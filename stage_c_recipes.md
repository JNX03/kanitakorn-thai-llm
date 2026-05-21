# Stage C Recipes — Apply After Stage 1 Eval

If Stage 1 (R1-Distill-14B + Thai SFT + EN mix) misses any top-3 target:

## Recipe 1: Math preservation (if MATH500 regresses below 82)

```bash
# Train math-only specialist on JUST R1 distill data + EN math
ssh ... 'cd /root/kanitakorn && \
  SFT_BASE=deepseek-ai/DeepSeek-R1-Distill-Qwen-14B \
  SFT_OUT=/root/kanitakorn/runs/specialist_math \
  SFT_EPOCHS=1 SFT_BS=4 SFT_GA=2 \
  SFT_MIX_EN=1 \
  CUDA_VISIBLE_DEVICES=0 \
  python3 train_2xa100_interruptible.py \
    --manifest dataset/sft_ready_r1d14b/manifest_math_only.json'

# Then merge with Stage 1:
python3 merge_sweep.py \
  --adapters stage1=runs/stage1_v1/final math=runs/specialist_math/final \
  --grid 0.7,0.3 0.6,0.4 0.5,0.5 \
  --base deepseek-ai/DeepSeek-R1-Distill-Qwen-14B \
  --eval-after
```

## Recipe 2: Thai boost (if THAIEXAM below 75)

```bash
# Train Thai-only specialist — high replication on OpenThaiEval/ThaiExam-style data
SFT_BASE=deepseek-ai/DeepSeek-R1-Distill-Qwen-14B \
SFT_OUT=/root/kanitakorn/runs/specialist_thai \
SFT_EPOCHS=2 SFT_BS=4 SFT_GA=2 SFT_MIX_EN=0 \
CUDA_VISIBLE_DEVICES=0 \
python3 train_2xa100_interruptible.py
```

## Recipe 3: DeepConf inference amplifier (FREE +20pp on AIME24)

Run after Stage 1 → essentially doubles AIME from base level:

```bash
ssh ... 'cd /root/kanitakorn && \
  python3 eval_deepconf.py \
    --base deepseek-ai/DeepSeek-R1-Distill-Qwen-14B \
    --adapter runs/stage1_v1/final \
    --benchmark aime24 --n 32 --drop-bottom-frac 0.25 \
    --out reports/v1_aime24_deepconf.json'
```

Note: DeepConf with n=32 takes ~30 min on AIME24 (30 problems × 32 samples × 8192 tokens).

## Recipe 4: Best-of-N with reward model

If AIME24 still below target after DeepConf — use a 7B reward model to rank samples:

```python
# pseudocode — would need to write
samples = model.sample(n=64)
rewards = reward_model(samples)
best = max(samples, key=lambda s: rewards[s])
```

Skipped for now (~$1-2 more in compute, complex).

## Recipe 5: Tool-use (SimpleTIR-style) for AIME

If still not at AIME 25+ — train a small wrapper that calls Python for arithmetic:
- arXiv:2509.02479 shows Qwen2.5-7B 22.1 → 50.5 with tool use
- Needs additional dataset of `<tool>python code</tool><output>...</output>` examples
- Out of scope for Run 1 — bookmark for Run 2

## Decision tree

```
Stage 1 done → eval top-3 (full sets)
├── all 3 beat target → ✅ done, run full 12-bench eval
├── MATH500 < 82 → Recipe 1 (math preservation merge)
├── THAIEXAM < 75 → Recipe 2 (Thai specialist)
├── AIME24 < 25 → Recipe 3 (DeepConf, free)
└── 2+ benchmarks miss → consider Run 2 with different hparams (LR, R, mix ratio)
```
