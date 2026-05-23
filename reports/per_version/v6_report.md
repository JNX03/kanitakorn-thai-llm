# Kanitakorn v6 Detailed Report

**Date trained:** 2026-05-22
**HF:** [Jnx03/kanitakorn-r1d-qwen14b-stage1-r1d-v6-rehearsal-20260522](https://huggingface.co/Jnx03/kanitakorn-r1d-qwen14b-stage1-r1d-v6-rehearsal-20260522)
**Base:** deepseek-ai/DeepSeek-R1-Distill-Qwen-14B
**Hardware:** 1× A40 (vast.ai $0.43/hr)
**Training time:** 57 min
**Cost:** ~$0.42

## Recipe
- **LoRA r=8**, alpha=16, target: attention + MLP layers
- **QLoRA 4-bit (nf4)** + bf16 compute
- **LR 2e-6**, cosine schedule, 2% warmup
- **Effective batch:** 2 × ga=4 = 8
- **Max seq len:** 2048
- **Epochs:** 0.5 (154 steps)
- **Dataset:** 5179 records weighted (30% thaiexam targeted, 25% math rehearsal, rest mix)

## Train Metrics
- **Final loss:** 1.487
- **Mean token accuracy:** 0.7203
- **Grad norm stable:** ~0.27
- Steady descent, no instability

## Full Benchmark Results
| Benchmark | Records | Method | Score | Target | Status |
|-----------|--------:|--------|------:|-------:|:------:|
| ThaiExam | 565 | Maj@4 | 62.3% (352/565) | >75 | ❌ -12.7pp |
| MATH500 | 500 | n=1, 4k tok | 62.0% (310/500) | >82 | ❌ -20pp |
| AIME24 | 30 | n=1, 4k tok | 23.33% (7/30) | >25 | ❌ -1.67pp |

## Comparison to Base
- Thai +3.3pp vs base 59% (rehearsal helped)
- Math stable
- AIME unchanged

## Lessons Learned
1. Rehearsal recipe (25% math mixed in) successfully preserved math reasoning
2. ThaiExam improvement small (+3pp) — needs different approach
3. r=8 may be too small for big Thai knowledge gains
