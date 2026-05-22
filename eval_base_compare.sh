#!/usr/bin/env bash
# Eval R1-Distill base (NO adapter) on same setup as v1/v2.
# Compares apples-to-apples whether SFT damage is real, or our setup undercounts.
#
# Expected (from DeepSeek paper, eval methodology may differ):
#   MATH500: ~93%
#   AIME24:  ~69%
#   ThaiExam: ~30-40% (no Thai SFT)
#
# Our same setup might score lower due to max_tokens=4096 limit + extractor.
set -e
cd /root/kanitakorn
[ -f .env.secrets ] && { set -a; source .env.secrets; set +a; }
export HF_HOME=/root/.hf
export CUDA_VISIBLE_DEVICES=0

BASE="${BASE:-deepseek-ai/DeepSeek-R1-Distill-Qwen-14B}"
TAG="${TAG:-base}"

mkdir -p reports
for bench in math500 aime24 thaiexam; do
    out="reports/${TAG}_${bench}.json"
    if [ -f "$out" ]; then echo "[skip] $out exists"; continue; fi
    case "$bench" in
        aime24) limit=30 ;;
        *) limit=100 ;;
    esac
    echo "=== [$(date)] BASE $bench (limit=$limit) ==="
    python3 eval_vllm.py \
        --base "$BASE" \
        --benchmark $bench --n 1 --limit $limit \
        --max-tokens 4096 --tp 1 \
        --out "$out" 2>&1 | tail -5
done

python3 -c "
import json, os
print('=== R1-Distill BASE (no SFT) vs Targets ===')
print(f\"{'Benchmark':12s}{'Base':>8s}{'v2':>8s}{'Target':>8s}\")
print('-'*45)
for b, t in [('math500',82),('aime24',25),('thaiexam',75)]:
    base_p = f'reports/${TAG}_{b}.json'
    v2_p = f'reports/v2q_{b}.json'
    base_v = json.load(open(base_p)).get('accuracy', 0)*100 if os.path.exists(base_p) else None
    v2_v = json.load(open(v2_p)).get('accuracy', 0)*100 if os.path.exists(v2_p) else None
    base_s = f'{base_v:.2f}' if base_v is not None else '-'
    v2_s = f'{v2_v:.2f}' if v2_v is not None else '-'
    print(f'{b:12s}{base_s:>8s}{v2_s:>8s}{t:>8}')
"
