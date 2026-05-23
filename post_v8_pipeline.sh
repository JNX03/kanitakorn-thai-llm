#!/usr/bin/env bash
# Auto-triggered after v8 training: push to HF + run full eval
set -e
cd /root/kanitakorn
set -a; source .env.secrets; set +a
export HF_HOME=/root/.hf
export CUDA_VISIBLE_DEVICES=0

# Wait for training done marker
while [ ! -d /root/kanitakorn/runs/r1d_kanitakorn_v8/final ]; do
    sleep 60
done
echo "[post-v8] training complete, pushing + evaluating at $(date)"

# Push to HF
python3 tools/hf_auto_publish.py \
    --adapter-dir runs/r1d_kanitakorn_v8/final \
    --base deepseek-ai/DeepSeek-R1-Distill-Qwen-14B \
    --stage stage1-r1d-v8-ddp \
    --hyper '{"lr":2e-6,"r":16,"epochs":0.7,"qlora":true,"ddp":2,"data":"v8 mega + Thai-CoT"}' \
    --notes "Kanitakorn v8: R1-Distill-14B 2xA40 DDP + Thai-CoT data. r=16 capacity boost."

# Full eval top-3
python3 eval_vllm.py --base deepseek-ai/DeepSeek-R1-Distill-Qwen-14B --adapter runs/r1d_kanitakorn_v8/final \
    --benchmark thaiexam --n 4 --limit 565 --max-tokens 2048 --max-model-len 8192 \
    --tp 1 --out reports/v8_thaiexam_full.json 2>&1 | tee v8_eval.log

python3 eval_vllm.py --base deepseek-ai/DeepSeek-R1-Distill-Qwen-14B --adapter runs/r1d_kanitakorn_v8/final \
    --benchmark math500 --n 1 --limit 500 --max-tokens 4096 --max-model-len 8192 \
    --tp 1 --out reports/v8_math500_full.json 2>&1 | tee -a v8_eval.log

python3 eval_vllm.py --base deepseek-ai/DeepSeek-R1-Distill-Qwen-14B --adapter runs/r1d_kanitakorn_v8/final \
    --benchmark aime24 --n 16 --limit 30 --max-tokens 4096 --max-model-len 8192 \
    --tp 1 --out reports/v8_aime24_maj16.json 2>&1 | tee -a v8_eval.log

echo "[post-v8] DONE at $(date)"
for f in reports/v8_*.json; do
    python3 -c "import json; d=json.load(open('$f')); print('$f:', round(d.get('accuracy',0)*100,2), '%')"
done
