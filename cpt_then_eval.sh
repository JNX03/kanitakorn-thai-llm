#!/usr/bin/env bash
# Pipeline: CPT → eval CPT alone → SFT on top of CPT → eval SFT+CPT
# Run after cpt_qwen3_thai.py finishes.
set -e
cd /root/kanitakorn
set -a; source .env.secrets; set +a
export HF_HOME=/root/.hf
export CUDA_VISIBLE_DEVICES=0

CPT_ADAPTER=/root/kanitakorn/runs/cpt_qwen3_thai/final
SFT_OUT=/root/kanitakorn/runs/cpt_then_sft

# Step 1: Push CPT adapter to HF
echo "=== [$(date)] STEP 1: Push CPT to HF ==="
python3 tools/hf_auto_publish.py \
    --adapter-dir "$CPT_ADAPTER" \
    --base Qwen/Qwen3-14B \
    --stage cpt-thai \
    --hyper '{"lr":1e-5,"steps":800,"r":16,"qlora":true,"corpus":"fineweb-2-tha_Thai","max_len":2048}' \
    --notes "Kanitakorn CPT v1: Qwen3-14B continued pretrain on Fineweb2-Thai"

# Step 2: Eval CPT alone on ThaiExam + MT-Bench
echo "=== [$(date)] STEP 2: Eval CPT alone ==="
python3 eval_vllm.py --base Qwen/Qwen3-14B --adapter "$CPT_ADAPTER" \
    --benchmark thaiexam --n 1 --limit 100 --max-tokens 2048 --max-model-len 8192 \
    --tp 1 --out reports/cpt_thaiexam.json 2>&1 | tee cpt_eval.log

python3 eval_vllm.py --base Qwen/Qwen3-14B --adapter "$CPT_ADAPTER" \
    --benchmark thaiexam --n 4 --limit 100 --max-tokens 2048 --max-model-len 8192 \
    --tp 1 --out reports/cpt_maj4_thaiexam.json 2>&1 | tee -a cpt_eval.log

python3 eval_ifeval_mtbench.py --base Qwen/Qwen3-14B \
    --bench mt_bench_th --limit 80 --max-tokens 2048 --max-model-len 8192 \
    --out reports/cpt_mt_bench_th.json 2>&1 | tee -a cpt_eval.log

echo "=== [$(date)] CPT eval done — check scores ==="
cat reports/cpt_*.json | python3 -c "
import sys, json
for line in sys.stdin:
    try:
        d = json.loads(line)
        print('-', d.get('benchmark'), round(d.get('accuracy',0)*100,2), '%')
    except: pass
"
echo "=== If CPT improved Thai >= base 67%, optionally run SFT-on-CPT next ==="
