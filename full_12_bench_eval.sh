#!/usr/bin/env bash
# Full 12-benchmark sweep — ONLY run if top-3 beat targets.
# Uses vLLM for speed. ~2 hours on single A100 for all benchmarks.
#
# Usage on instance:
#   ADAPTER=runs/stage1_v1/final TAG=v1 bash full_12_bench_eval.sh
set -e

cd /root/kanitakorn
[ -f .env.secrets ] && { set -a; source .env.secrets; set +a; }
export HF_HOME=/root/.hf
export CUDA_VISIBLE_DEVICES=0

ADAPTER="${ADAPTER:-runs/stage1_v1/final}"
BASE="${BASE:-deepseek-ai/DeepSeek-R1-Distill-Qwen-14B}"
TAG="${TAG:-v1}"

mkdir -p reports

# Note: benchmarks supported by eval_self_consistency.py / eval_vllm.py
# All 12 targets:
declare -A BENCHES=(
    ["aime24"]="math-ai/aime24 / integer / 30 records"
    ["aime25"]="math-ai/aime25 / integer / 30 records (proxy for AIME24-TH)"
    ["math500"]="HuggingFaceH4/MATH-500 / math / 500 records"
    ["thaiexam"]="scb10x/thai_exam / mcq / 565 records (5 configs)"
    ["openthaieval"]="iapp/openthaieval / mcq / 1232 records"
    ["ifeval_th"]="typhoon-ai/ifeval-th / 215 records"
    ["mt_bench_th"]="ThaiLLM-Leaderboard/mt-bench-thai / 91 records"
)

# Eval each benchmark (limit appropriately)
for bench in aime24 aime25 math500 thaiexam openthaieval; do
    out="reports/full_${TAG}_${bench}.json"
    if [ -f "$out" ]; then
        echo "[skip] $out exists"
        continue
    fi
    case "$bench" in
        aime24|aime25)  limit=30 ;;
        math500) limit=500 ;;
        thaiexam) limit=565 ;;
        openthaieval) limit=300 ;; # subset for speed
        *) limit=200 ;;
    esac
    echo "=== [$(date)] $bench (limit=$limit) ==="
    python3 eval_vllm.py \
        --base "$BASE" --adapter "$ADAPTER" \
        --benchmark $bench --n 1 --limit $limit \
        --max-tokens 4096 --tp 1 \
        --out "$out" 2>&1 | tail -5
done

# Print summary table
python3 -c "
import json, os
targets = {
    'aime24': 25, 'aime25': 15, 'math500': 82, 'thaiexam': 75, 'openthaieval': 80,
}
print(f\"{'Benchmark':20s}{'Score':>10s}{'Target':>10s}  Beat?\")
print('-'*55)
scores = {}
for b, t in targets.items():
    p = f'reports/full_${TAG}_{b}.json'
    if not os.path.exists(p):
        print(f'{b:20s}{\"—\":>10s}{t:>10}   ?')
        continue
    d = json.load(open(p))
    v = d.get('accuracy', 0)
    v_pct = v*100 if v <= 1 else v
    mark = '✅' if v_pct > t else '❌'
    print(f'{b:20s}{v_pct:>10.2f}{t:>10}   {mark}')
    scores[b] = v_pct
json.dump(scores, open('reports/full_${TAG}_summary.json','w'), indent=2)
"

echo "=== [$(date)] DONE ==="
