"""
Speculative decoding with a small draft: simulated from MEASURED inputs only.
  agreement  results/2026-09-18/draft_accept_qwen3_0.6b.json -- host/draft_accept.cpp run with Qwen3-0.6B-Q8_0
             teacher-forced on the phone's greedy Qwen3-30B-A3B output (bmoe_stack base_rep1a; the tool's token
             count reproduced the engine's exactly: 34 prompt + 256 generated, prompt_prefix_ok)
  verify     claims verify_cost_n2/n3/n5: verifying k positions costs 1.71 / 2.37 / 3.71 single decodes
Greedy verification accepts a draft chain exactly as far as the agreement holds consecutively, so the per-token
array is simulated directly (no geometric assumption). The draft's own cost is NOT measured here; the best case
(free draft) is reported, which is an upper bound on any real speedup.
Run:
  python moe-phone/gates/draft_accept_sim.py --agree results/2026-09-18/draft_accept_qwen3_0.6b.json \
      --out results/2026-09-18/draft_accept_sim.json
"""
import argparse, json, random
ap = argparse.ArgumentParser(); ap.add_argument("--agree", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
d = json.load(open(a.agree)); m = d["match"]
if not d.get("prompt_prefix_ok"): raise SystemExit("tokenization boundary unreliable")
COST = {2: 1.71, 3: 2.37, 5: 3.71}
def tokens_per_verify(seq, L):
    s = t = v = 0
    while s < len(seq):
        k = 0
        while k < L and s + k < len(seq) and seq[s + k] == 1: k += 1
        t += k + 1; v += 1; s += k + 1
    return t / v
rows = [dict(k=k, draft_len=k - 1, tokens_per_verify=tokens_per_verify(m, k - 1), verify_cost=c,
             speedup_free_draft=tokens_per_verify(m, k - 1) / c) for k, c in COST.items()]
random.seed(0); B = 16; blocks = [m[i:i + B] for i in range(0, len(m), B)]; boot = []
for _ in range(2000):
    s = sum((random.choice(blocks) for _ in blocks), []); boot.append(sum(s) / len(s))
boot.sort()
out = dict(n_gen=d["n_gen"], agreement=sum(m) / len(m), agreement_ci95=[boot[50], boot[1950]], rows=rows,
           best_speedup_free_draft=max(r["speedup_free_draft"] for r in rows),
           breakeven_agreement_k2=COST[2] - 1.0)
json.dump(out, open(a.out, "w"), indent=1); print(json.dumps(out, indent=1))
