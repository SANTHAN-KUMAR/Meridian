// draft_accept.cpp — would a small draft model make speculative decoding pay on the phone?
//
// Teacher-forced agreement: feed the DRAFT model the target's exact greedy output (prompt in the model's chat
// template + the generated text), evaluate every position in one pass, and record whether the draft's argmax
// at position i-1 equals the target's token i. Under greedy verification, a draft chain started at position s
// is accepted exactly as far as this agreement holds consecutively from s (while it holds, the draft's own
// prefix IS the target's prefix), so the per-position array is sufficient to simulate any draft length.
//
// Output (stdout, one JSON object): n_prompt, n_gen, match[] (0/1 per generated token), and a tokenization
// check (the prompt's tokens must be a prefix of the full sequence's tokens, else the boundary is unreliable).
//
// Build (host, against the BigMoeOnEdge host build of llama.cpp):
//   g++ -O2 -std=c++17 draft_accept.cpp -I$E/third_party/llama.cpp/include -I$E/third_party/llama.cpp/ggml/include \
//       -L$E/build-host/bin -lllama -lggml -lggml-base -Wl,-rpath,$E/build-host/bin -o draft_accept
// Run:
//   draft_accept MODEL.gguf PROMPT_FILE GENERATED_TEXT_FILE
#include "llama.h"
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

static std::string slurp(const char * p) {
    std::ifstream f(p, std::ios::binary);
    std::stringstream ss; ss << f.rdbuf(); return ss.str();
}

static std::vector<llama_token> tok(const llama_vocab * v, const std::string & s) {
    int n = -llama_tokenize(v, s.c_str(), (int) s.size(), nullptr, 0, /*add_special*/ false, /*parse_special*/ true);
    std::vector<llama_token> t(n);
    if (llama_tokenize(v, s.c_str(), (int) s.size(), t.data(), n, false, true) != n) { t.clear(); }
    return t;
}

int main(int argc, char ** argv) {
    if (argc != 4) { std::fprintf(stderr, "usage: %s MODEL PROMPT_FILE GEN_FILE\n", argv[0]); return 2; }
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) { std::fprintf(stderr, "load failed\n"); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    // The engine wraps the prompt with the model's own chat template (thinking on); reproduce it.
    const std::string user = slurp(argv[2]);
    llama_chat_message msg{"user", user.c_str()};
    const char * tmpl = llama_model_chat_template(model, nullptr);
    std::vector<char> buf(user.size() * 2 + 4096);
    int nb = llama_chat_apply_template(tmpl, &msg, 1, /*add_ass*/ true, buf.data(), (int) buf.size());
    if (nb < 0) { std::fprintf(stderr, "chat template failed\n"); return 1; }
    const std::string prompt(buf.data(), nb);
    const std::string gen = slurp(argv[3]);

    std::vector<llama_token> tp = tok(vocab, prompt);
    std::vector<llama_token> tf = tok(vocab, prompt + gen);
    bool prefix_ok = tp.size() <= tf.size();
    for (size_t i = 0; prefix_ok && i < tp.size(); ++i) prefix_ok = tp[i] == tf[i];
    const int n_prompt = (int) tp.size(), n_all = (int) tf.size();

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n_all + 16; cp.n_batch = n_all + 16; cp.n_ubatch = n_all + 16;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { std::fprintf(stderr, "context failed\n"); return 1; }
    llama_batch b = llama_batch_init(n_all, 0, 1);
    for (int i = 0; i < n_all; ++i) {
        b.token[i] = tf[i]; b.pos[i] = i; b.n_seq_id[i] = 1; b.seq_id[i][0] = 0;
        b.logits[i] = (i >= n_prompt - 1) ? 1 : 0;   // logits at every position that predicts a generated token
    }
    b.n_tokens = n_all;
    if (llama_decode(ctx, b) != 0) { std::fprintf(stderr, "decode failed\n"); return 1; }
    const int n_vocab = llama_vocab_n_tokens(vocab);
    std::printf("{\"n_prompt\": %d, \"n_gen\": %d, \"prompt_prefix_ok\": %s, \"match\": [", n_prompt, n_all - n_prompt,
                prefix_ok ? "true" : "false");
    for (int i = n_prompt; i < n_all; ++i) {
        const float * lg = llama_get_logits_ith(ctx, i - 1);
        int best = 0;
        for (int v = 1; v < n_vocab; ++v) if (lg[v] > lg[best]) best = v;
        std::printf("%s%d", i == n_prompt ? "" : ",", best == tf[i] ? 1 : 0);
    }
    std::printf("]}\n");
    llama_batch_free(b); llama_free(ctx); llama_model_free(model); llama_backend_free();
    return 0;
}
