# Resume Protocol — When Vast.ai SSH Returns

Always check this first if you lost context. Goal anchor: `memory/project_thai_llm_campaign_goal.md`.

## Current campaign state (2026-05-21)

**Instance details** (last known good):
- Direct: `ssh -p 30840 root@194.228.55.129`
- Proxy: `ssh -p 38215 root@ssh4.vast.ai`
- Path: `/root/kanitakorn` (NOT /workspace — that mount doesn't exist on this image)
- 2× A100 SXM4 40GB confirmed
- Deps installed: torch 2.6.0+cu124, transformers 5.9.0, peft 0.19.1, trl 1.4.0, bnb 0.49.2

**Base model decision**: `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B` (15B params, user-approved exception)
- Prebaked scores: MATH500=93.9, AIME24=69.7 (Cons@64 80.0)
- License: MIT
- Memory: requires QLoRA 4-bit (set `SFT_QLORA=1` or auto-detected for 14B+ bases)

**Training script**: `train_2xa100_interruptible.py` — DDP-aware, QLoRA-aware, auto-resume from latest checkpoint
**Launch**: `bash sync_and_launch.sh <port> <host> [smoke|bake|full]`
**HF auto-push**: `python3 tools/hf_auto_publish.py --adapter-dir <path> --base <base> --stage <stage> --eval-json <results>`

## Step-by-step when SSH returns

### A. Smoke test (5 min, confirms install + model load)
```bash
bash sync_and_launch.sh 30840 194.228.55.129 smoke
ssh -p 30840 root@194.228.55.129 'tmux attach -t smoke'
# Expect: model loads in 4-bit, generates "1+1 = \boxed{2}", uses ~15GB per GPU
```

### B. Baseline eval (run before training — establish floor) (~30 min)
```bash
ssh -p 30840 root@194.228.55.129 'tmux new -d -s base "
  cd /root/kanitakorn
  set -a; source .env.secrets; set +a   # loads HF_TOKEN, OPENROUTER_API_KEY
  export HF_HOME=/root/.hf
  python3 eval_self_consistency.py --base deepseek-ai/DeepSeek-R1-Distill-Qwen-14B --benchmark thaiexam --n 1 --limit 100 --out reports/base_thaiexam.json 2>&1 | tee /tmp/base_thaiexam.log
  python3 eval_self_consistency.py --base deepseek-ai/DeepSeek-R1-Distill-Qwen-14B --benchmark math500 --n 1 --limit 100 --out reports/base_math500.json 2>&1 | tee /tmp/base_math500.log
  python3 eval_self_consistency.py --base deepseek-ai/DeepSeek-R1-Distill-Qwen-14B --benchmark aime24 --n 1 --limit 30 --out reports/base_aime24.json 2>&1 | tee /tmp/base_aime24.log
"'
```

### C. Decision tree based on baseline

| Baseline result | Action |
|-----------------|--------|
| All top-3 already beat targets | Skip training, push base + go straight to Stage D (DeepConf eval @64) |
| MATH+AIME beat, THAIEXAM weak | Run **light Thai SFT** (1 epoch, LR=5e-6, LoRA r=8) to add Thai without breaking math |
| All weak | Full Stage B (3 epochs, LR=1e-5, LoRA r=16) |

### D. Stage B full training launch
```bash
SFT_BASE=deepseek-ai/DeepSeek-R1-Distill-Qwen-14B \\
SFT_EPOCHS=2 \\
SFT_LORA_R=16 \\
bash sync_and_launch.sh 30840 194.228.55.129 full
```

Monitor: `ssh -p 30840 root@194.228.55.129 'tail -F /tmp/train.log'`

### E. Post-train eval + HF push
```bash
ssh -p 30840 root@194.228.55.129 'cd /root/kanitakorn && \
  python3 eval_self_consistency.py --base deepseek-ai/DeepSeek-R1-Distill-Qwen-14B --adapter runs/stage1_v1/final --benchmark thaiexam --n 1 --limit 100 --out reports/v1_thaiexam.json && \
  python3 eval_self_consistency.py --base deepseek-ai/DeepSeek-R1-Distill-Qwen-14B --adapter runs/stage1_v1/final --benchmark math500 --n 1 --limit 100 --out reports/v1_math500.json && \
  python3 eval_self_consistency.py --base deepseek-ai/DeepSeek-R1-Distill-Qwen-14B --adapter runs/stage1_v1/final --benchmark aime24 --n 1 --limit 30 --out reports/v1_aime24.json'

# Then push to HF:
ssh -p 30840 root@194.228.55.129 'cd /root/kanitakorn && \
  python3 tools/hf_auto_publish.py \\
    --adapter-dir runs/stage1_v1/final \\
    --base deepseek-ai/DeepSeek-R1-Distill-Qwen-14B \\
    --stage stage1 \\
    --hyper "{\\"lr\\":1e-5,\\"epochs\\":2,\\"r\\":16,\\"qlora\\":true,\\"mix\\":\\"30:70\\"}" \\
    --eval-json reports/v1_thaiexam.json \\
    --notes "First Stage B run"'
```

## Local background jobs in progress (as of 22:58)

| PID | Job | Output | Progress |
|----:|-----|--------|---------|
| 10692 | codex hotpot scale (stuck?) | `dataset/train/train_hotpotqa_agentic_codex*.jsonl` | 0 records (probably wedged) |
| 41720 | codex teacher_loop scale | `train_teacher_loop_th_codex_20260521_222923.jsonl` | 105 records (may have ended) |
| 27196 | distill_from_teacher AIME-TH | `train_distill_r1_aime_th.jsonl` | ~40/200 |
| 28120 | distill_from_teacher MATH500-TH | `train_distill_r1_math500_th.jsonl` | ~96/150 |

When these finish, **re-package the manifest** to include the new distill data:
```bash
cd C:/Users/Jnx03/Desktop/kanitakornv2
python3 tools/package_and_verify.py --include-distill  # if such a flag exists; else manual concat
# Or just append distill into existing aime_th_train.jsonl / math500_th_train.jsonl
```

## Important credentials (stored in `.env.secrets`, gitignored)

Tokens are in `.env.secrets` (HF_TOKEN, OPENROUTER_API_KEY). Source it before running anything:
```bash
set -a; source .env.secrets; set +a
```

SSH public key: `ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIMtHLY6tMZ/PMrVqwCLoAswOg0znCKF4dYr9PghB5Hac Jnx03@DESKTOP-B2J3RCM`

## Targets to beat (top-3 first, then rest)

| Benchmark | Target | Strategy |
|-----------|-------:|----------|
| THAIEXAM | >75 | Thai SFT (current 663+ records) + R1 distillation for math-style |
| MATH500 | >82 | Base R1-Distill already 93.9 — just preserve |
| AIME24 | >25 | Base R1-Distill already 69.7 — just preserve |

If all top-3 beat after Stage B, **expand to full 12-benchmark eval** (only then).

## Open questions (no need to wait for user answers)

1. Confirmation that distillation data should be in `train_distill_*.jsonl` and merged into manifest before training, OR trained on as a 9th family. **Default action**: merge into existing aime_th/math500_th files at 5-10% weight.

2. Whether to push every intermediate checkpoint or just notable ones. User said "every notable adapter". **Default**: push at end of each stage.

3. Whether MT-Bench scoring uses Thai or English judge. Both papers use English judges; Typhoon uses Claude. **Default**: gemini-2.5-flash-lite as judge.
