// route_trace — collect a model's OWN routing decisions through llama.cpp.
//
// Why (moe-phone POSITION.md §11, S9): the geometry-transfer test needs routing
// traces for a model whose 16-bit weights do not fit any GPU available to this
// project (Qwen3-30B-A3B). llama.cpp runs it from a mmap'd GGUF on CPU, and its
// graph names every MoE layer's selected experts "ffn_moe_topk-<layer>"
// (src/llama-graph.cpp, build_moe_ffn: cb(selected_experts, "ffn_moe_topk", il)).
// This tool asks the scheduler's eval callback for exactly those tensors and
// copies them out: I32 [n_expert_used, n_tokens] per layer per ubatch.
//
// It does NOT tokenize: token ids come from a file written by
// gates/llamacpp_traces.py with the model's HF tokenizer, using the same corpus
// and windowing as gates/traces_sparsity.py. Identical ids mean a difference in
// routing is a difference in the model/format/engine, not in the text.
//
// Each window is an independent sequence (memory cleared, positions 0..W-1),
// decoded as ONE batch, i.e. teacher forcing — the same as the HF forward pass
// that produced the committed OLMoE trace.
//
// Output: binary file
//   char[8] "MOETRC01"; int32 n_layers, top_k, n_windows, window_len, n_experts
//   int32 layer_ids[n_layers]
//   int16 experts[n_windows * window_len][n_layers][top_k]   (token-major)
//
// Build (after building llama.cpp in build-cpu):
//   g++ -O2 -std=c++17 route_trace.cpp -I$LL/include -I$LL/ggml/include \
//       -L$LL/build-cpu/bin -lllama -lggml -lggml-base -Wl,-rpath,$LL/build-cpu/bin -o route_trace
// Run:
//   route_trace MODEL.gguf TOKENS.bin OUT.bin [threads]
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

struct Ctx {
    int n_tokens_in_batch = 0;
    int top_k = -1;
    std::map<int, std::vector<int32_t>> by_layer;   // layer -> [n_tokens * k] for current batch
    long n_seen = 0;
};

static bool cb_eval(struct ggml_tensor * t, bool ask, void * ud) {
    Ctx * c = (Ctx *) ud;
    const char * pfx = "ffn_moe_topk-";
    const bool want = strncmp(t->name, pfx, strlen(pfx)) == 0;
    if (ask) return want;
    if (!want) return true;
    if (t->type != GGML_TYPE_I32) {
        fprintf(stderr, "FATAL: %s has type %d, expected I32\n", t->name, (int) t->type);
        exit(3);
    }
    const int layer = atoi(t->name + strlen(pfx));
    const int k = (int) t->ne[0];
    const int n = (int) t->ne[1];
    if (c->top_k < 0) c->top_k = k;
    if (k != c->top_k) { fprintf(stderr, "FATAL: top_k changed %d -> %d\n", c->top_k, k); exit(3); }
    std::vector<int32_t> buf((size_t) k * n);
    if (t->nb[0] != sizeof(int32_t) || t->nb[1] != (size_t) k * sizeof(int32_t)) {
        fprintf(stderr, "FATAL: %s is not contiguous (nb0=%zu nb1=%zu)\n", t->name, t->nb[0], t->nb[1]);
        exit(3);
    }
    ggml_backend_tensor_get(t, buf.data(), 0, buf.size() * sizeof(int32_t));
    auto & v = c->by_layer[layer];
    v.insert(v.end(), buf.begin(), buf.end());      // ubatches append in token order
    c->n_seen++;
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: route_trace MODEL.gguf TOKENS.bin OUT.bin [threads]\n");
        return 2;
    }
    const char * model_path = argv[1];
    const int n_threads = argc > 4 ? atoi(argv[4]) : 8;

    // tokens file: int32 n_windows, int32 window_len, int32 ids[n_windows*window_len]
    FILE * ft = fopen(argv[2], "rb");
    if (!ft) { perror(argv[2]); return 1; }
    int32_t nw = 0, wl = 0;
    if (fread(&nw, 4, 1, ft) != 1 || fread(&wl, 4, 1, ft) != 1) { fprintf(stderr, "bad tokens header\n"); return 1; }
    std::vector<llama_token> ids((size_t) nw * wl);
    if (fread(ids.data(), 4, ids.size(), ft) != ids.size()) { fprintf(stderr, "short tokens file\n"); return 1; }
    fclose(ft);

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;   // default load mode (mmap) keeps a 17 GB GGUF off the heap
    llama_model * model = llama_model_load_from_file(model_path, mp);
    if (!model) { fprintf(stderr, "failed to load %s\n", model_path); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    for (auto id : ids) if (id < 0 || id >= n_vocab) { fprintf(stderr, "token id %d outside vocab %d\n", id, n_vocab); return 1; }

    Ctx c;
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = wl;
    cp.n_batch = wl;
    cp.n_ubatch = wl;
    cp.n_threads = n_threads;
    cp.n_threads_batch = n_threads;
    cp.cb_eval = cb_eval;
    cp.cb_eval_user_data = &c;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "failed to create context\n"); return 1; }

    FILE * fo = nullptr;
    std::vector<int> layers;
    std::vector<int16_t> row;
    for (int w = 0; w < nw; w++) {
        llama_memory_clear(llama_get_memory(ctx), true);
        c.by_layer.clear();
        llama_batch b = llama_batch_get_one(ids.data() + (size_t) w * wl, wl);
        if (llama_decode(ctx, b) != 0) { fprintf(stderr, "decode failed at window %d\n", w); return 1; }
        if (c.by_layer.empty()) { fprintf(stderr, "FATAL: no ffn_moe_topk tensors seen — not an MoE graph?\n"); return 3; }
        if (layers.empty()) {
            for (auto & kv : c.by_layer) layers.push_back(kv.first);
            fo = fopen(argv[3], "wb");
            if (!fo) { perror(argv[3]); return 1; }
            // expert count from GGUF metadata ("<arch>.expert_count"); 0 if absent,
            // in which case the converter requires --num-experts
            int32_t n_expert = 0;
            char key[256], val[256];
            for (int32_t i = 0; i < llama_model_meta_count(model); i++) {
                llama_model_meta_key_by_index(model, i, key, sizeof key);
                const size_t kl = strlen(key), sl = strlen(".expert_count");
                if (kl > sl && strcmp(key + kl - sl, ".expert_count") == 0) {
                    llama_model_meta_val_str_by_index(model, i, val, sizeof val);
                    n_expert = atoi(val);
                }
            }
            int32_t hdr[5] = {(int32_t) layers.size(), c.top_k, nw, wl, n_expert};
            fwrite("MOETRC01", 1, 8, fo);
            fwrite(hdr, 4, 5, fo);
            std::vector<int32_t> lid(layers.begin(), layers.end());
            fwrite(lid.data(), 4, lid.size(), fo);
            fprintf(stderr, "%zu MoE layers, top_k %d, n_expert %d\n", layers.size(), c.top_k, hdr[4]);
        }
        const int k = c.top_k, L = (int) layers.size();
        for (int li = 0; li < L; li++) {
            auto it = c.by_layer.find(layers[li]);
            if (it == c.by_layer.end() || it->second.size() != (size_t) k * wl) {
                fprintf(stderr, "FATAL: layer %d captured %zu ids, expected %d\n", layers[li],
                        it == c.by_layer.end() ? (size_t) 0 : it->second.size(), k * wl);
                return 3;
            }
        }
        row.assign((size_t) wl * L * k, 0);
        for (int li = 0; li < L; li++) {
            const auto & v = c.by_layer[layers[li]];
            for (int tkn = 0; tkn < wl; tkn++)
                for (int j = 0; j < k; j++)
                    row[((size_t) tkn * L + li) * k + j] = (int16_t) v[(size_t) tkn * k + j];
        }
        fwrite(row.data(), sizeof(int16_t), row.size(), fo);
        fflush(fo);
        fprintf(stderr, "window %d/%d done\n", w + 1, nw);
    }
    fclose(fo);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
