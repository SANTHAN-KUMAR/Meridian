#define private public
#include "/tmp/claude-1000/bmoe-prefetch/core/src/moe/router_hook.h"
#undef private
#include <cstdio>
#include <vector>
#include <cmath>
using namespace bmoe;
static int fails = 0;
#define CHECK(c, msg) do { if (!(c)) { std::printf("FAIL %s\n", msg); ++fails; } else std::printf("ok   %s\n", msg); } while (0)
int main() {
    // 16 experts, top-4. Scores: experts 0..3 top (10, 9, 8, 7.2), boundary expert 4 at 7.0, rest 0.
    std::vector<float> s(16, 0.0f);
    s[0] = 10; s[1] = 9; s[2] = 8; s[3] = 7.2f; s[4] = 7.0f;
    std::vector<uint8_t> res(16, 0); // route_miss == 0: everything is a miss
    std::vector<int32_t> spec, keep;
    RouterHook::GateCounts g;
    // 1. gates off == old behaviour: top spec_max misses in rank order
    RouterHook::build_spec_lists(s, 4, 0.0f, 3, res, spec, keep);
    CHECK(spec == std::vector<int32_t>({0, 1, 2}) && keep.empty(), "gates off: first 3 predicted misses");
    // 2. rank gate R=2: only ranks 0,1
    g = {}; RouterHook::build_spec_lists(s, 4, 0.0f, 3, res, spec, keep, 2, -1.0f, &g);
    CHECK(spec == std::vector<int32_t>({0, 1}), "rank<2 keeps ranks 0,1");
    CHECK(g.misses == 4 && g.cut_rank == 2 && g.cut_cap == 0, "rank counters: 4 misses, 2 cut");
    // 3. margin gate: sd of scores; expert 3 is 0.2 above boundary -> small z, cut; 0,1,2 pass at M=0.2
    double m = 0, v = 0; for (float x : s) m += x; m /= 16; for (float x : s) v += (x - m) * (x - m); double sd = std::sqrt(v / 16);
    g = {}; RouterHook::build_spec_lists(s, 4, 0.0f, 8, res, spec, keep, 0, (float) (0.5 / sd), &g);
    CHECK(spec == std::vector<int32_t>({0, 1, 2}) && g.cut_margin == 1, "margin cuts the expert at the top-k edge only");
    // 4. residents are kept, never gated
    res[1] = 1; g = {};
    RouterHook::build_spec_lists(s, 4, 0.0f, 8, res, spec, keep, 1, -1.0f, &g);
    CHECK(keep == std::vector<int32_t>({1}) && spec == std::vector<int32_t>({0}) && g.misses == 3 && g.cut_rank == 2,
          "resident retained regardless of gate; rank gate applies to misses");
    // 5. cap still applies after the gates
    res.assign(16, 0); g = {};
    RouterHook::build_spec_lists(s, 4, 0.0f, 1, res, spec, keep, 0, -1.0f, &g);
    CHECK(spec.size() == 1 && g.cut_cap == 3, "spec-max cap counted");
    std::printf(fails ? "%d FAILED\n" : "all passed\n", fails);
    return fails != 0;
}
