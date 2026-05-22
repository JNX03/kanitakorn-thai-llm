#!/usr/bin/env bash
set -e
cd /root/kanitakorn
set -a; source .env.secrets; set +a
export HF_HOME=/root/.hf
export CUDA_VISIBLE_DEVICES=0

# Push first to HF
python3 tools/hf_auto_publish.py \
    --adapter-dir runs/qwen3_sft/final \
    --base Qwen/Qwen3-14B \
    --stage stage1-qwen3 \
    --hyper '{"lr":5e-6,"epochs":0.5,"r":8,"qlora":true,"thai_total":2500,"bs":2,"ga":4}' \
    --notes "Qwen3-14B + Thai SFT (v3): light recipe to add Thai capability" 2>&1 | tee push_qwen3.log

# Then eval
python3 eval_vllm.py --base Qwen/Qwen3-14B --adapter runs/qwen3_sft/final \
    --benchmark thaiexam --n 1 --limit 100 --max-tokens 2048 --max-model-len 8192 \
    --tp 1 --out reports/v3qwen3_thaiexam.json 2>&1 | tee qwen3_eval.log

python3 eval_vllm.py --base Qwen/Qwen3-14B --adapter runs/qwen3_sft/final \
    --benchmark thaiexam --n 4 --limit 100 --max-tokens 2048 --max-model-len 8192 \
    --tp 1 --out reports/v3qwen3_maj4_thaiexam.json 2>&1 | tee -a qwen3_eval.log

python3 eval_ifeval_mtbench.py --base Qwen/Qwen3-14B --bench mt_bench_th \
    --limit 80 --max-tokens 2048 --max-model-len 8192 \
    --out reports/v3qwen3_mt_bench_th.json 2>&1 | tee -a qwen3_eval.log

echo "DONE"
