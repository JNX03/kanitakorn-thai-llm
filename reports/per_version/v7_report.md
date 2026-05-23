# Kanitakorn v7 Detailed Report

**Date trained:** 2026-05-22
**HF:** [Jnx03/kanitakorn-r1d-qwen14b-stage1-r1d-v7-mega-20260522](https://huggingface.co/Jnx03/kanitakorn-r1d-qwen14b-stage1-r1d-v7-mega-20260522)
**Base:** deepseek-ai/DeepSeek-R1-Distill-Qwen-14B
**Hardware:** 1× A40 (vast.ai $0.43/hr)
**Training time:** 70 min
**Cost:** ~$0.51

## Recipe — same as v6 but MORE data
- **LoRA r=8** (same as v6)
- **QLoRA 4-bit nf4** + bf16
- **LR 2e-6**, cosine
- **Effective batch:** 2 × ga=4 = 8
- **Max seq len:** 2048
- **Epochs:** 0.7 (216 steps, +40% vs v6)
- **Dataset:** 10918 unique Thai SFT (2x v6) — emphasis on synth corpus
- **Weights:** 45% thaiexam targeted, 28% math/code rehearsal, 27% other

## Train Metrics
- **Final loss:** 1.364 (better than v6's 1.487)
- **Mean token accuracy:** 0.7404
- **Grad norm stable:** ~0.30

## Full Benchmark Results
| Benchmark | Records | Method | Score | Target | Status |
|-----------|--------:|--------|------:|-------:|:------:|
| ThaiExam | 565 | Maj@4 | 61.77% (349/565) | >75 | ❌ -13.2pp |
| MATH500 | 500 | n=1, 4k tok | 63.4% (317/500) | >82 | ❌ -18.6pp |
| MATH500 | 200 | Maj@8 | 68.0% (136/200) | >82 | ❌ -14pp |
| AIME24 | 30 | n=1, 4k tok | 23.33% (7/30) | >25 | ❌ -1.67pp |
| AIME24 | 30 | **Maj@16** | **46.67%** (14/30) | >25 | ✅ **BEAT +21.67pp** |

## Key Discovery: Self-Consistency Decisive for AIME
**Same model, n=1 → Maj@16: +23.34pp** (23.33 → 46.67%). This pattern matches R1-Distill paper claims (paper: pass@1=69.7, cons@64=80.0). Our 14B model has the math reasoning — just needs sampling to surface it.

For MATH500 Maj@8 adds +4.6pp (63 → 68). Less dramatic than AIME — math reasoning more deterministic.

For ThaiExam Maj@4 → Maj@8 plateau (61.77% both). Knowledge bottleneck, not noise.

## Failure Analysis (ThaiExam)
By category (heuristic):
| Category | Score | N | Note |
|----------|------:|--:|------|
| math | 85.7% | 28 | strong (preserved from base) |
| social studies | 83.3% | 6 | small N |
| literature | 65.7% | 35 | mid |
| english | 55.6% | 9 | mid |
| biology | 57.1% | 7 | mid |
| history | 42.9% | 7 | **WEAK** |
| investment | 41.2% | 17 | **WEAK** |
| grammar | 0% | 2 | **WEAK (tiny N)** |

**Critical insight:** Model REASONS IN ENGLISH/CHINESE when answering Thai questions. Saw "Alright,", "好的" prefixes. Thai-CoT data needed.

## Lessons Learned
1. **More data alone doesn't help ThaiExam** — v7 with 2x data got same as v6
2. **Self-consistency at inference is the killer feature** for math — should ship Maj@N as part of inference recipe
3. **Need targeted Thai-language reasoning data** to fix English-reasoning bug
4. **History + Investment are weak categories** — generate more there
5. **Plateau at r=8** — may need r=16 next
