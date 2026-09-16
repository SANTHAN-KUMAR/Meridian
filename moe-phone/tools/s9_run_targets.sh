#!/bin/bash
# s9_run_targets.sh — collect and score the S9 targets whose registrations are committed.
#
# For each target, in order: wait for its GGUF download to finish (DL_DONE in the log
# written by the download job), tokenize wikitext-2 test with the model's own tokenizer
# (gates/llamacpp_traces.py, the traces_sparsity procedure), collect routing through
# llama.cpp (tools/route_trace.cpp), convert, build the f_crit curve (cache_sim), and score
# against the pre-registration (gates/s9_score.py). Targets run one at a time: the largest
# GGUF exceeds the laptop's RAM, so parallel traces would measure paging, not routing.
# A failed target is logged and the next one still runs.
#
# Usage: s9_run_targets.sh MOE_WORK_DIR REPO_DIR
set -u
W=$1; REPO=$2
G="$REPO/moe-phone/gates"; D="$REPO/moe-phone/results/2026-09-16"; P="$W/venv/bin/python"
CONF="$D/cache_fcrit_OLMoE-1B-7B-0924-q4_0-llamacpp.json"
run_target() {   # name dl_log gguf hf_tokenizer tag
  local name=$1 dlog=$2 gguf=$3 tok=$4 tag=$5 L="$W/s9_$1.log"
  echo "=== $name: waiting for download $(date -Iseconds)" > "$L"
  until grep -q DL_DONE "$W/$dlog" 2>/dev/null; do sleep 60; done
  echo "download done $(date -Iseconds)" >> "$L"
  ( cd "$W" \
    && "$P" "$G/llamacpp_traces.py" tokens --model "$tok" --out "tok_$name.bin" \
    && "$W/route_trace" "models/$gguf" "tok_$name.bin" "$name.trc" 20 2>&1 | tail -3 \
    && test -s "$name.trc" \
    && "$P" "$G/llamacpp_traces.py" convert "$name.trc" --tag "$tag" --gguf "models/$gguf" --llama-commit 9f31776 --out-dir "$D" \
    && "$P" "$G/cache_sim.py" "$D/traces_$tag.npz" --fractions-around-crit --out-name "cache_fcrit_$name" --out-dir "$D" | tail -2 \
    && "$P" "$G/s9_score.py" --prereg "$D/s9_prereg_$name.json" --target-curve "$D/cache_fcrit_$name.json" --confound-curve "$CONF" --out-dir "$D" \
    && echo "TARGET_OK $name" ) >> "$L" 2>&1 || echo "TARGET_FAILED $name" >> "$L"
}
run_target granite-3.1-3b-a800m dl_granite3b.log granite-3.1-3b-a800m-instruct-Q4_0.gguf ibm-granite/granite-3.1-3b-a800m-instruct granite-3.1-3b-a800m-q4_0-llamacpp
# within-family score uses the same curve; no engine confound between two llama.cpp Q4_0 traces
"$P" "$G/s9_score.py" --prereg "$D/s9_prereg_granite-3.1-3b-a800m_within-family.json" \
     --target-curve "$D/cache_fcrit_granite-3.1-3b-a800m.json" --out-dir "$D" >> "$W/s9_granite-3.1-3b-a800m.log" 2>&1 \
  && echo "WITHIN_FAMILY_OK" >> "$W/s9_granite-3.1-3b-a800m.log"
run_target gpt-oss-20b dl_gptoss.log gpt-oss-20b-MXFP4.gguf openai/gpt-oss-20b gpt-oss-20b-mxfp4-llamacpp
run_target Qwen3-30B-A3B dl_qwen3.log Qwen3-30B-A3B-Q4_0.gguf Qwen/Qwen3-30B-A3B Qwen3-30B-A3B-q4_0-llamacpp
echo "ALL_TARGETS_DONE $(date -Iseconds)" >> "$W/s9_all.log"
