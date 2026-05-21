# Run 2 Plan — Apply if v1b Still Shows Catastrophic Forgetting

## Diagnosis from Run 1

Run 1 results (with buggy extractor):
- MATH500: 52% (R1-Distill base 93.9%)
- THAIEXAM: 37%

If v1b re-eval (with fixed extractor) still shows MATH500 < 80, our training **damaged** R1-Distill's math reasoning.

## Run 2 hyperparameters (anti-forgetting)

| Param | Run 1 | Run 2 | Rationale |
|-------|------:|------:|-----------|
| LR | 1e-5 | **2e-6** | 5× lower — R1-Distill is delicate |
| LoRA r | 16 | **8** | Smaller change |
| Epochs | 1 | **0.5** | Light touch |
| Per-device BS | 4 | 4 | Same |
| Grad accum | 2 | 2 | Same (eff batch 8) |
| Max length | 1536 | **2048** | More room for reasoning |
| EN mix ratio | 60% (6k of 10.5k) | **80%** (heavy EN) | Preserve base capability |
| EN math (MetaMath) | 2000 | **8000** | More math anchor |
| EN code (OpenCoder) | 2000 | 2000 | Same |
| EN chat (OpenHermes) | 2000 | 2000 | Same |

## Data adjustments

1. **Fix misrouted records**: OpenThaiEval R1-distill traces went into math500_th. Re-route to openthaieval family.
2. **Drop teacher_loop_th**: 410 multi-turn records with verifier feedback may confuse training.
3. **Wrap distill with `<think>...</think>` tags**: this preserves R1's expected output format.

## Launch command (Run 2)

```bash
ssh -p 30840 root@194.228.55.129 'cd /root/kanitakorn
SFT_OUT=/root/kanitakorn/runs/stage1_v2 \
SFT_EPOCHS=0.5 SFT_BS=4 SFT_GA=2 \
SFT_MAX_LEN=2048 SFT_SAVE_STEPS=100 \
SFT_LORA_R=8 SFT_LR_OVERRIDE=2e-6 \
CUDA_VISIBLE_DEVICES=0 \
bash boot_and_train.sh'
```

Need to modify train_2xa100_interruptible.py to:
1. Accept SFT_LR_OVERRIDE env var
2. Tune EN mix to 80% (or reduce Thai sample to ~1500)

## Decision: if v1b STILL bad after fixes

Run 2 with above hparams. Expected outcome:
- MATH500 should stay ≥ 85 (mostly base preserved)
- THAIEXAM 40-50 (modest gain)
- AIME24 50+ (mostly base preserved)

If Run 2 also fails, fallback options:
- Switch base to R1-Distill-7B (smaller, less destructive)
- Use unsloth fast path (different optimizer, may help)
- Direct adapter scaling at inference (no retrain needed)
