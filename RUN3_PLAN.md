# Run 3 Plan — Pivot Strategy

## Learning from Run 1 + Run 2

**Run 1 (LR 1e-5, r=16, 1 epoch):** Catastrophic forgetting. math500=55, aime24=23, thaiexam=36 — all below target.

**Run 2 (LR 2e-6, r=8, 0.3 epoch):** Light touch — expect mostly base preserved, minimal Thai gain.

If Run 2 also doesn't beat targets, the SFT approach on R1-Distill-Qwen-14B is fundamentally limited for ThaiExam. We need a different angle.

## Pivot options (rank-ordered by expected impact)

### Option A: Use R1-Distill BASE + Cons@N (no SFT)
- Pre-tested by DeepSeek paper: R1-Distill-14B Cons@64 = 80% AIME24 (vs 69.7 single)
- Expected: math500 ~96, aime24 ~80, thaiexam ~30-40 (no Thai SFT)
- **Beats math + aime targets, fails thaiexam**
- Script: `eval_base_consN.sh` (pre-written)

### Option B: Switch base to ThaiLLM-8B + Light SFT
- ThaiLLM-8B base ThaiExam = 48% → with our SFT possibly 55-60%
- ThaiLLM-8B base math = ~30-40 (Qwen3-8B class)
- Expected: thaiexam ~55, math ~40, aime ~15 — fails 2 targets
- Doesn't solve the problem

### Option C: Two-model ensemble (best of both)
- Use R1-Distill-14B for math/aime
- Use ThaiLLM-8B-SFT for thaiexam
- Route by benchmark
- **Cheap to implement, technically wins ALL benchmarks separately**
- Not a single unified model — may not count as "one Thai LLM"

### Option D: Scale-down v1 or v2 adapter
- `scale_adapter.py` builds α-scaled version
- If v2 hurt math, try v2 at α=0.5 (half-strength)
- Inference cost: same as base + adapter
- May rescue both math and Thai with right alpha
- **Cheap to test (~5 min × 4 alpha values)**

## Recommended order if Run 2 fails

1. Run Option D first (alpha sweep) — 30 min on instance
2. If still not beating: Option A (cons@N base) for math/aime
3. If user wants single model: Option D + DeepConf at inference

## Estimated budget left
- Started: $9.62
- Now: ~$6
- Per eval cycle (3 benchmarks vLLM): ~$0.30
- Per training run (~30 min): ~$0.20
- Plenty of budget for 5+ more experiments

## Honest reality

Hitting all 12 targets on a single 14B base is extremely hard:
- R1-Distill-14B can hit math (93+) and aime (69+) but lacks Thai
- ThaiLLM-8B can hit thaiexam (50+) but lacks reasoning
- No single base hits ALL targets

Partial victory: beat math + aime + IFEval (English ones) → ~5/12 targets. Document and ship.

For full victory: would need a 14B model pretrained on BOTH math reasoning AND Thai (doesn't exist publicly).
