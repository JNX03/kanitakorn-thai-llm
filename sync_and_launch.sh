#!/usr/bin/env bash
# One-command: sync code, install deps if needed, launch training in tmux.
# Run from local Windows machine. Requires ssh access to vast.ai instance.
#
# Usage:
#   bash sync_and_launch.sh <port> <host> [stage]
#     stage: smoke (default), bake, full
#
# Examples:
#   bash sync_and_launch.sh 30840 194.228.55.129 smoke
#   bash sync_and_launch.sh 30840 194.228.55.129 full
set -e

PORT="${1:?usage: $0 <port> <host> [stage]}"
HOST="${2:?usage: $0 <port> <host> [stage]}"
STAGE="${3:-smoke}"
SSH="ssh -p $PORT -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null root@$HOST"
SCP="scp -P $PORT -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"

HF_TOKEN="${HF_TOKEN:?must set HF_TOKEN env var}"
OPENROUTER_API_KEY="${OPENROUTER_API_KEY:-}"

echo "=== [1/4] packing local artifacts ==="
cd "$(dirname "$0")"
tar --exclude='__pycache__' --exclude='*.pyc' --exclude='.git' --exclude='runs' \
    --exclude='*.log' --exclude='_locked_snapshot' --exclude='.claude' \
    -czf /tmp/kt-sync.tar.gz \
    train_2xa100_interruptible.py train_unsloth_fast.py fire_2xa100.sh \
    eval_self_consistency.py eval_deepconf.py ab_base_compare.py \
    CAMPAIGN_7DAY.md VERSION_LOG.md \
    tools dataset/sft_ready

echo "=== [2/4] uploading to instance ==="
$SCP /tmp/kt-sync.tar.gz root@$HOST:/root/
$SSH "mkdir -p /root/kanitakorn && cd /root/kanitakorn && tar xzf /root/kt-sync.tar.gz && ls -la"

echo "=== [3/4] ensuring deps ==="
$SSH "python3 -c 'import torch, transformers, peft, trl, bitsandbytes, datasets' 2>&1 | tail -3 || \
      pip install --break-system-packages -q --upgrade transformers peft trl datasets safetensors huggingface_hub accelerate bitsandbytes sentencepiece scipy jsonschema sentence-transformers requests"

echo "=== [4/4] launching $STAGE ==="
case "$STAGE" in
  smoke)
    $SSH "tmux kill-session -t smoke 2>/dev/null; tmux new -d -s smoke \"
      cd /root/kanitakorn
      export HF_TOKEN='$HF_TOKEN'
      export OPENROUTER_API_KEY='$OPENROUTER_API_KEY'
      export HF_HOME=/root/.hf
      echo === [\\\$(date)] SMOKE: download base + 1-record forward pass ===
      python3 -c \\\"
import os, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
base = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-14B'
print('loading', base)
tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
mdl = AutoModelForCausalLM.from_pretrained(base, quantization_config=bnb, device_map='auto', trust_remote_code=True, attn_implementation='sdpa')
print('OK. mem GB:', sum(torch.cuda.memory_allocated(i)/1e9 for i in range(torch.cuda.device_count())))
inp = tok('What is 1+1? Answer with \\\\\\\\boxed{}.', return_tensors='pt').to(mdl.device)
out = mdl.generate(**inp, max_new_tokens=128, do_sample=False)
print('out:', tok.decode(out[0]))
\\\" 2>&1 | tail -30
      echo === [\\\$(date)] SMOKE DONE ===
    \" ; tmux ls"
    ;;
  bake)
    $SSH "tmux kill-session -t bake 2>/dev/null; tmux new -d -s bake \"
      cd /root/kanitakorn
      export HF_TOKEN='$HF_TOKEN'
      export HF_HOME=/root/.hf
      python3 ab_base_compare.py --bases deepseek-ai/DeepSeek-R1-Distill-Qwen-14B deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --benchmarks thaiexam math500 aime24 --subset-n 500 --limit-per-bench 50 --out-dir runs/bakeoff 2>&1 | tee /tmp/bake.log
    \" ; tmux ls"
    ;;
  full)
    $SSH "tmux kill-session -t train 2>/dev/null; tmux new -d -s train \"
      cd /root/kanitakorn
      export HF_TOKEN='$HF_TOKEN'
      export OPENROUTER_API_KEY='$OPENROUTER_API_KEY'
      export HF_HOME=/root/.hf
      export SFT_BASE='${SFT_BASE:-deepseek-ai/DeepSeek-R1-Distill-Qwen-14B}'
      export SFT_OUT='${SFT_OUT:-/root/kanitakorn/runs/stage1_v1}'
      export SFT_EPOCHS='${SFT_EPOCHS:-2}'
      export SFT_LORA_R='${SFT_LORA_R:-16}'
      export SFT_MIX_EN='${SFT_MIX_EN:-1}'
      export SFT_QLORA='${SFT_QLORA:-1}'
      export SFT_PROJECT=/root/kanitakorn
      while true; do
        echo \\\"[\\\$(date)] starting training\\\"
        NGPU=\\\$(nvidia-smi -L | wc -l)
        if [ \\\$NGPU -ge 2 ]; then
          torchrun --nproc_per_node=\\\$NGPU --master_port=29501 train_2xa100_interruptible.py 2>&1 | tee -a /tmp/train.log
        else
          python3 train_2xa100_interruptible.py 2>&1 | tee -a /tmp/train.log
        fi
        ec=\\\${PIPESTATUS[0]}
        if [ \\\$ec -eq 0 ]; then
          echo \\\"[\\\$(date)] training COMPLETE\\\"; break
        fi
        echo \\\"[\\\$(date)] training failed ec=\\\$ec, retry in 60s\\\"
        sleep 60
      done
    \" ; tmux ls"
    ;;
  *)
    echo "Unknown stage: $STAGE (use smoke|bake|full)"; exit 1
    ;;
esac

echo ""
echo "=== launched. monitor: ==="
echo "  $SSH 'tmux attach -t $STAGE'"
echo "  $SSH 'tail -F /tmp/train.log'"
