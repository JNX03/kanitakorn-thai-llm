#!/usr/bin/env bash
# Force full re-eval of top-3 benchmarks on FULL test sets with max tokens.
# Use after training to get publishable scores.
#
# Usage on remote:
#   ADAPTER=/root/kanitakorn/runs/stage1_v1/final TAG=v1 bash eval_top3_full.sh
#   # or for base:
#   ADAPTER= TAG=base bash eval_top3_full.sh
set -e

cd /root/kanitakorn
export HF_TOKEN="${HF_TOKEN:?must set HF_TOKEN}"
export HF_HOME=/root/.hf
ADAPTER="${ADAPTER:-}"
TAG="${TAG:-base}"
BASE="${BASE:-deepseek-ai/DeepSeek-R1-Distill-Qwen-14B}"

mkdir -p reports

adapter_flag=""
if [ -n "$ADAPTER" ]; then
  adapter_flag="--adapter $ADAPTER"
fi

# Full test sets, with self-consistency n=1 (single sample, greedy-ish)
for bench in math500 aime24 thaiexam; do
  out="reports/full_${TAG}_${bench}.json"
  if [ -f "$out" ]; then
    echo "[skip] $out exists"
    continue
  fi
  echo "=== [$(date)] eval $bench on full set ==="
  case "$bench" in
    aime24)  limit=30 ;;   # full AIME = 30 problems
    math500) limit=500 ;;  # full MATH-500
    thaiexam) limit=565 ;; # full ThaiExam (across 5 configs)
  esac
  python3 eval_self_consistency.py \
    --base "$BASE" $adapter_flag \
    --benchmark $bench --n 1 --limit $limit \
    --out "$out"
done

# Summary
python3 -c "
import json, os
scores = {}
for b in ['thaiexam', 'math500', 'aime24']:
    p = 'reports/full_${TAG}_' + b + '.json'
    if os.path.exists(p):
        d = json.load(open(p))
        scores[b] = d.get('accuracy')
print('=== FULL EVAL ${TAG} ===')
for b, v in scores.items():
    print(f'  {b:12s}: {v}')
"

echo "[done] $(date)"
