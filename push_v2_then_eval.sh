#!/usr/bin/env bash
# RUN IMMEDIATELY WHEN INSTANCE COMES BACK.
# 1. Push v2 adapter to HF (preserves work before any new preemption)
# 2. Then eval
set -e

cd /root/kanitakorn
set -a; source .env.secrets; set +a
export HF_HOME=/root/.hf
export CUDA_VISIBLE_DEVICES=0

# Step 1: PUSH FIRST
echo "=== [$(date)] PUSH v2 to HF ==="
python3 tools/hf_auto_publish.py \
    --adapter-dir runs/stage1_v2/final \
    --base deepseek-ai/DeepSeek-R1-Distill-Qwen-14B \
    --stage stage1-v2 \
    --hyper '{"lr":2e-6,"epochs":0.3,"r":8,"qlora":true,"mix":"30:70","bs":4,"ga":2,"max_len":1536,"thai_total":1500}' \
    --notes "Run 2 anti-forgetting: LR 2e-6, r=8, 0.3 epochs, Thai cap 1500"

# Step 2: vLLM eval (math500 + aime24 + thaiexam)
echo "=== [$(date)] EVAL v2 ==="
ADAPTER=/root/kanitakorn/runs/stage1_v2/final TAG=v2 STAGE=stage1-v2 bash post_train_pipeline.sh

echo "=== [$(date)] DONE ==="
