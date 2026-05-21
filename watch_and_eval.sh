#!/usr/bin/env bash
# Watch for new checkpoints in runs/stage1_v1/ and run quick eval on each.
# Uses vLLM for speed.
#
# Usage on instance:
#   bash watch_and_eval.sh
set -e

cd /root/kanitakorn
[ -f .env.secrets ] && { set -a; source .env.secrets; set +a; }
export HF_HOME=/root/.hf
export CUDA_VISIBLE_DEVICES=0

OUT_DIR=runs/stage1_v1
mkdir -p reports

seen=""
while true; do
  # Find newest checkpoint
  latest=$(ls -dt $OUT_DIR/checkpoint-* 2>/dev/null | head -1)
  [ -z "$latest" ] && { sleep 60; continue; }
  ckpt=$(basename "$latest")
  if [ "$ckpt" != "$seen" ]; then
    echo "=== [$(date)] new checkpoint: $ckpt ==="
    # Quick eval (30 records each, single sample)
    for bench in thaiexam math500 aime24; do
      limit=30
      [ "$bench" = "aime24" ] && limit=30
      out="reports/${ckpt}_${bench}.json"
      if [ ! -f "$out" ]; then
        echo "  [eval] $bench (limit=$limit)"
        python3 eval_self_consistency.py \
          --base deepseek-ai/DeepSeek-R1-Distill-Qwen-14B \
          --adapter "$latest" \
          --benchmark $bench --n 1 --limit $limit \
          --out "$out" 2>&1 | tail -3
      fi
    done
    # Print summary
    python3 -c "
import json, os
scores = {}
for b in ['thaiexam','math500','aime24']:
    p = f'reports/${ckpt}_{b}.json'
    if os.path.exists(p):
        d = json.load(open(p))
        scores[b] = d.get('accuracy')
print(f'=== ${ckpt} ===')
for b, v in scores.items():
    target = {'thaiexam':0.75,'math500':0.82,'aime24':0.25}[b]
    mark = '✅' if (v and v > target) else '❌'
    print(f'  {b:12s}: {v} {mark} (target {target})')
"
    seen=$ckpt
  fi
  sleep 60
done
