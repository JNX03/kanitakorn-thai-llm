#!/usr/bin/env bash
set -e
cd /root/kanitakorn
set -a; source .env.secrets; set +a

python3 tools/hf_auto_publish.py \
    --adapter-dir runs/stage1_v1/final \
    --base deepseek-ai/DeepSeek-R1-Distill-Qwen-14B \
    --stage stage1-v1-exp \
    --hyper '{"lr":1e-5,"epochs":1,"r":16,"qlora":true,"mix":"30:70","bs":4,"ga":2,"max_len":1536}' \
    --notes "Stage 1 experimental: extractor bugs suspected, see VERSION_LOG"
