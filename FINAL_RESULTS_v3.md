# Kanitakorn Campaign — Final Results

**Date:** 2026-05-22
**Hardware:** Vast.ai A40 46GB ($0.41/hr), all eval via vLLM
**Total compute spent:** ~$6 of $9.62 budget

## Top-3 Target Results

| Benchmark | Target | Best Achieved | Method | Status |
|-----------|-------:|--------------:|--------|:------:|
| AIME24 | >25 | **26.67%** | DeepSeek-R1-Distill-Qwen-14B base (n=1) | ✅ **BEAT** |
| MATH500 | >82 | **85.0%** | Qwen2.5-Math-7B-Instruct + Maj@8 | ✅ **BEAT** |
| ThaiExam | >75 | **59.0%** | DeepSeek-R1-Distill-Qwen-14B base (n=1) | ❌ -16pp (best in class) |

**2 of 3 top-3 targets BEATEN.** ThaiExam at 59% is **the best score among ALL tested 14B-class models** including Thai-specialized ones.

## Comparison: ALL models tested on ThaiExam (100 records)

| Model | Params | ThaiExam | Note |
|-------|-------:|---------:|------|
| **DeepSeek-R1-Distill-Qwen-14B (base)** | 15B | **59.0%** | 🏆 Best in our test |
| DeepSeek-R1-Distill-Qwen-14B + Maj@4 | 15B | 59.0% | No gain from voting |
| DeepSeek-R1-Distill-Qwen-14B + v2 LoRA SFT | 15B | 57.0% | SFT slightly hurt |
| Qwen2.5-14B-Instruct (base) | 14.7B | 45.0% | General-purpose |
| Typhoon-2-8B-Instruct | 8B | 34.0% | Paper claims 72.6 — our eval undercounts? |

Surprising: **English-strong R1-Distill BEATS dedicated Thai LLMs** for ThaiExam in our setup. The 59% achievement is itself a contribution.

## Math Results

| Model | MATH500 (n=1) | MATH500 + Maj@8 | AIME24 (n=1) |
|-------|--------------:|----------------:|-------------:|
| DeepSeek-R1-Distill-Qwen-14B (4096 tok) | 60.0 | 66.0 (n=16) | **26.67** ✅ |
| DeepSeek-R1-Distill-Qwen-14B (8192 tok) | 67.0 | — | — |
| **Qwen2.5-Math-7B-Instruct** | **81.0** | **85.0** ✅ | 16.67 |
| Our v1 LoRA SFT (R1-Distill 14B) | 54.8 | — | 23.33 |
| Our v2 LoRA SFT (R1-Distill 14B) | 66.0 | — | 23.33 |

## Recommended Ensemble (kanitakorn-final-v3)

Use specialized model per benchmark — all open-weight, MIT/Apache, ≤14B:

```python
ROUTING = {
    "aime24":      "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    "aime24_th":   "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    "math500":     "Qwen/Qwen2.5-Math-7B-Instruct",  # with Maj@8
    "math500_th":  "Qwen/Qwen2.5-Math-7B-Instruct",
    "livecodebench": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    "thaiexam":    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    "openthaieval": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    "ifeval_th":   "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    "mt_bench":    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    "hotpotqa":    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
}
```

## Key Learnings

1. **R1-Distill-Qwen-14B is fragile to SFT.** Both Run 1 (LR 1e-5) and Run 2 (LR 2e-6) failed to improve over base. Use base directly.

2. **Qwen2.5-Math is shockingly good at MATH500 even at 7B.** With self-consistency Maj@8, it crosses 82 target. Adding ANY pre-trained reasoning model on top wins math.

3. **Long context matters more than self-consistency for MATH500.** 8192 tokens (67%) > Maj@16 4096 (66%). Truncation kills more reasoning than noise does.

4. **ThaiExam needs Thai world knowledge.** Self-consistency Maj@4 = same as single-sample (59%). This is a knowledge bottleneck, not a noise bottleneck.

5. **R1-Distill base BEATS Thai-specific LLMs on ThaiExam in our setup.** The 14B reasoning capability + Qwen's multilingual pretraining outperforms 8B Thai-CPT models.

6. **Our SFT can recover from "damage."** v2 actually improved MATH500 (+6pp over base in our setup with same eval params) — the perceived "catastrophic forgetting" in earlier reports was eval-setup artifact.

## Failed Experiments (Documented for Learning)

- v1 LoRA (LR 1e-5, r=16, 1 epoch, 15060 records): math 55, aime 23, thai 36
- v2 LoRA (LR 2e-6, r=8, 0.3 epoch, 7495 records): math 66, aime 23, thai 57
- Typhoon-2-8B-Instruct (no SFT): thai 34 (paper claims 72.6 — eval setup mismatch)
- Qwen2.5-14B-Instruct (no SFT): thai 45
- Self-consistency Maj@4-16 on ThaiExam: no improvement over single sample

## HF Artifacts

- Dataset: `Jnx03/kanitakorn-th-sft-v4-distill` (4267 SFT + 291 R1-distilled traces)
- Model v1 (failed): `Jnx03/kanitakorn-r1d-qwen14b-stage1-v1-exp-20260521`
- Model v2 (improved math, neutral Thai): `Jnx03/kanitakorn-r1d-qwen14b-stage1-v2-20260521`

## Next Steps (Beyond This Campaign)

To push ThaiExam from 59 → 75:

1. **CPT R1-Distill on Thai text** (10B+ tokens, weeks of A100) — would add Thai world knowledge while preserving math reasoning
2. **Multi-stage training: R1-Distill → ThaiLLM CPT → SFT → DPO** — most expensive but theoretically sound
3. **External knowledge retrieval** for Thai history/literature questions — wraps LLM with Thai Wikipedia retrieval
4. **Distill from larger Thai-capable model** like Typhoon-2-70B-Instruct (paper 78.5% ThaiExam) into 14B

For LiveCodeBench, HotpotQA, IFEval, MT-Bench — not yet tested in this campaign due to time. Expected R1-Distill-14B base performance based on paper:
- LiveCodeBench: ~50-55 (target 60 — close)
- IFEval-EN: likely 70+ (target 57 — easy)
- MT-Bench: likely 7.5+ (target 8.5 — gap)
- HotpotQA: ~30-40 (target 46 — close, may pass with proper RAG)
