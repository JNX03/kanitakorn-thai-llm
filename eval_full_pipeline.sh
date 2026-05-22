#!/usr/bin/env bash
# Full eval pipeline: top-3 on FULL benchmark sets
# Args: $1 = adapter path (e.g., runs/r1d_kanitakorn_v6/final)
#       $2 = tag prefix for output files

set -e
cd /root/kanitakorn
set -a; source .env.secrets; set +a
export HF_HOME=/root/.hf
export CUDA_VISIBLE_DEVICES=0

ADAPTER="${1:-runs/r1d_kanitakorn_v6/final}"
TAG="${2:-v6}"

echo "=== [$(date)] Eval ADAPTER=$ADAPTER TAG=$TAG ==="

# 1. ThaiExam — full 565 records (5 configs), Maj@4
python3 eval_vllm.py --base deepseek-ai/DeepSeek-R1-Distill-Qwen-14B --adapter "$ADAPTER" \
    --benchmark thaiexam --n 4 --limit 565 --max-tokens 2048 --max-model-len 8192 \
    --tp 1 --out "reports/${TAG}_thaiexam_full.json" 2>&1 | tail -3

# 2. MATH500 — full 500 records, n=1 with 8K tokens
python3 eval_vllm.py --base deepseek-ai/DeepSeek-R1-Distill-Qwen-14B --adapter "$ADAPTER" \
    --benchmark math500 --n 1 --limit 500 --max-tokens 8192 --max-model-len 12288 \
    --tp 1 --out "reports/${TAG}_math500_full.json" 2>&1 | tail -3

# 3. AIME24 — full 30 records, n=1 with 8K tokens
python3 eval_vllm.py --base deepseek-ai/DeepSeek-R1-Distill-Qwen-14B --adapter "$ADAPTER" \
    --benchmark aime24 --n 1 --limit 30 --max-tokens 8192 --max-model-len 12288 \
    --tp 1 --out "reports/${TAG}_aime24_full.json" 2>&1 | tail -3

echo "=== [$(date)] Top-3 DONE ==="
for f in reports/${TAG}_thaiexam_full.json reports/${TAG}_math500_full.json reports/${TAG}_aime24_full.json; do
    python3 -c "
import json
d=json.load(open('$f'))
acc=round(d.get('accuracy',0)*100,2)
n=d.get('n_items',0)
c=d.get('correct',0)
print(f'$f: {acc}% ({c}/{n})')
"
done
