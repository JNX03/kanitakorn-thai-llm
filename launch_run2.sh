#!/usr/bin/env bash
# Launch Run 2 with anti-forgetting hparams.
#
# Hyper-changes vs Run 1:
#   LR: 1e-5 → 2e-6 (5× lower)
#   LoRA r: 16 → 8 (smaller change)
#   Epochs: 1 → 0.3 (light touch)
#   Thai records cap: 4528 → 1500 (less aggressive)
#   EN math: 2000 → 8000 records (heavier preservation)
#
# Run on instance:
#   bash /root/kanitakorn/launch_run2.sh
set -e
cd /root/kanitakorn
set -a; source .env.secrets; set +a

# Clean prior state to re-run training (skip baseline; trust prior)
rm -rf /root/kanitakorn/runs/stage1_v2
rm -f /root/kanitakorn/_state/train_done /root/kanitakorn/_state/eval_done /root/kanitakorn/_state/push_done
mkdir -p /root/kanitakorn/_state
touch /root/kanitakorn/_state/baseline_ok  # skip baseline

tmux kill-session -t pipeline2 2>/dev/null
tmux new -d -s pipeline2 "
cd /root/kanitakorn
set -a; source .env.secrets; set +a
export CUDA_VISIBLE_DEVICES=0
export SFT_BASE=deepseek-ai/DeepSeek-R1-Distill-Qwen-14B
export SFT_OUT=/root/kanitakorn/runs/stage1_v2
export SFT_EPOCHS=0.3
export SFT_BS=4
export SFT_GA=2
export SFT_MAX_LEN=2048
export SFT_SAVE_STEPS=100
export SFT_LORA_R=8
export SFT_LR=2e-6
export SFT_QLORA=1
export SFT_PROJECT=/root/kanitakorn
export SFT_MANIFEST=/root/kanitakorn/dataset/sft_ready_r1d14b/manifest.json
export SFT_THAI_TOTAL=1500
# Disable EN mix would be too risky — keep but with NoOpenHermes for cleaner math/code mix
bash boot_and_train.sh 2>&1 | tee pipeline_v2.log
"
sleep 3
tmux ls
echo ""
echo "Monitor: ssh -p 30840 root@194.228.55.129 'tail -F /root/kanitakorn/pipeline_v2.log'"
