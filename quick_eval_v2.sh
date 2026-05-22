#!/usr/bin/env bash
# Quick eval: ~15 min total (vs 50 min full). Get directional signal before next preemption.
# 100 records each for math500/thaiexam, 30 for aime24.
set -e
cd /root/kanitakorn
[ -f .env.secrets ] && { set -a; source .env.secrets; set +a; }
export HF_HOME=/root/.hf
export CUDA_VISIBLE_DEVICES=0

ADAPTER="${ADAPTER:-runs/stage1_v2/final}"
BASE="${BASE:-deepseek-ai/DeepSeek-R1-Distill-Qwen-14B}"
TAG="${TAG:-v2q}"  # q = quick

mkdir -p reports

for bench in math500 aime24 thaiexam; do
    out="reports/${TAG}_${bench}.json"
    if [ -f "$out" ]; then echo "[skip] $out exists"; continue; fi
    case "$bench" in
        aime24)  limit=30 ;;
        *) limit=100 ;;
    esac
    echo "=== [$(date)] $bench (limit=$limit) ==="
    python3 eval_vllm.py \
        --base "$BASE" --adapter "$ADAPTER" \
        --benchmark $bench --n 1 --limit $limit \
        --max-tokens 4096 --tp 1 \
        --out "$out" 2>&1 | tail -5
done

python3 -c "
import json, os
print(f'=== ${TAG} QUICK EVAL ===')
print(f\"{'Benchmark':12s}{'Score':>10s}{'Target':>10s}  Beat?\")
print('-'*45)
for b, t in [('math500', 82), ('aime24', 25), ('thaiexam', 75)]:
    p = f'reports/${TAG}_{b}.json'
    if not os.path.exists(p):
        print(f'{b:12s}{\"—\":>10s}{t:>10}   ?')
        continue
    d = json.load(open(p))
    v = d.get('accuracy', 0) * 100
    mark = '✅' if v > t else '❌'
    print(f'{b:12s}{v:>10.2f}{t:>10}   {mark}')
"
