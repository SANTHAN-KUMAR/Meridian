// repack_gguf.cpp — rewrite a GGUF's Q4_0 expert slices in ggml-cpu's repacked (interleaved) layout, IN PLACE in a copy.
//
// Why: --repack-experts repacks every streamed slice on the I/O lane after its read. On the phone that moved ~15 ms of
// compute into ~17 ms of stall (smoke bmoe_repack_smoke_20260919_0844). The repack is a same-size byte permutation, so the
// file can carry the repacked bytes from the start. The engine then only tags the tensors (--experts-prerepacked) and
// never repacks at run time. The header, tensor types and offsets are unchanged; only expert slice bytes differ.
// The layout depends on the CPU features ggml-cpu detects (q4_0_4x8 on i8mm), so this MUST run on the target device
// with the engine's own libggml-cpu. It writes <file>.repacked, which records the chosen layout per tensor, the counts and
// the file size. The engine refuses --experts-prerepacked without a matching marker.
//   repack_gguf <copy.gguf>   (the copy is modified in place; cp the original first)
#include "ggml.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <fcntl.h>
#include <unistd.h>

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: repack_gguf <copy.gguf>\n"); return 2; }
    const char * path = argv[1];
    ggml_context * meta = nullptr;
    gguf_init_params gp = { /*no_alloc=*/true, &meta };
    gguf_context * g = gguf_init_from_file(path, gp);
    if (!g || !meta) { std::fprintf(stderr, "FATAL cannot read gguf %s\n", path); return 1; }
    const size_t data_off = gguf_get_data_offset(g);
    const int n = (int) gguf_get_n_tensors(g);
    const int fd = ::open(path, O_RDWR | O_CLOEXEC);
    if (fd < 0) { std::perror("open"); return 1; }
    // a tensor object with real (host) data is needed only for its shape/type; bind tags it for the repack traits
    ggml_init_params ip = { (size_t) 64 * 1024 * 1024, nullptr, true };
    ggml_context * work = ggml_init(ip);
    std::vector<char> buf;
    int n_rep = 0, n_skip = 0, n_fail = 0;
    long long slices = 0;
    std::string marker;
    for (int i = 0; i < n; ++i) {
        const char * name = gguf_get_tensor_name(g, i);
        if (!std::strstr(name, "_exps.weight")) continue;
        ggml_tensor * mt = ggml_get_tensor(meta, name);
        if (!mt || mt->type != GGML_TYPE_Q4_0) { ++n_skip; marker += std::string("skip ") + name + " " + (mt ? ggml_type_name(mt->type) : "?") + "\n"; continue; }
        ggml_tensor * t = ggml_new_tensor_3d(work, mt->type, mt->ne[0], mt->ne[1], mt->ne[2]);
        ggml_set_name(t, name);
        if (!ggml_cpu_repack_bind_tensor(t)) { ++n_skip; marker += std::string("skip ") + name + " no-repacked-form\n"; continue; }
        const size_t slice = (size_t) mt->nb[2];
        const size_t off = data_off + gguf_get_tensor_offset(g, i);
        buf.resize(slice);
        bool ok = true;
        for (int64_t e = 0; e < mt->ne[2] && ok; ++e) {
            const off_t o = (off_t) (off + (size_t) e * slice);
            if (::pread(fd, buf.data(), slice, o) != (ssize_t) slice) { ok = false; break; }
            if (!ggml_cpu_repack_slice(t, buf.data(), slice)) { ok = false; break; }
            if (::pwrite(fd, buf.data(), slice, o) != (ssize_t) slice) { ok = false; break; }
            ++slices;
        }
        if (!ok) { ++n_fail; std::fprintf(stderr, "FATAL %s failed; the copy is now inconsistent, delete it\n", name); break; }
        ++n_rep;
        marker += std::string("repacked ") + name + "\n";
        if (n_rep % 24 == 0) std::fprintf(stderr, "repack_gguf: %d tensors\n", n_rep);
    }
    ::fsync(fd);
    const off_t size = ::lseek(fd, 0, SEEK_END);
    ::close(fd);
    if (n_fail) return 1;
    const std::string mpath = std::string(path) + ".repacked";
    FILE * m = std::fopen(mpath.c_str(), "w");
    if (!m) { std::perror("marker"); return 1; }
    std::fprintf(m, "repack_gguf v1\nfile_size %lld\nrepacked_tensors %d\nskipped_tensors %d\nslices %lld\n%s", (long long) size, n_rep,
                 n_skip, slices, marker.c_str());
    std::fclose(m);
    std::printf("RESULT repacked_tensors=%d skipped=%d slices=%lld file_size=%lld marker=%s\n", n_rep, n_skip, slices, (long long) size,
                mpath.c_str());
    gguf_free(g);
    ggml_free(meta);
    ggml_free(work);
    return 0;
}
