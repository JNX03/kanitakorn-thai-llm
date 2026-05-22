# Kanitakorn Campaign Results — Run 1 + Run 2

**Date:** 2026-05-22. **Status:** Targets NOT met. Documented for next iteration.

## Targets vs Results

| Benchmark | Target | Run 1 (v1) | Run 2 (v2) | R1-Distill base (ref) |
|-----------|-------:|----------:|-----------:|----------------------:|
| MATH500 | >82 | **54.8** ❌ | **66.0** ❌ | 93.9 (paper) |
| AIME24 | >25 | **23.3** ❌ | **23.3** ❌ | 69.7 (paper, ours undercounts) |
| THAIEXAM | >75 | **35.9** ❌ | preempted | n/a |
| (9 others) | various | not tested | not tested | n/a |

**0 / 3 top-3 targets beaten in either run.**

## Recipes tried

### Run 1: Aggressive SFT
- Base: DeepSeek-R1-Distill-Qwen-14B (15B, MIT)
- QLoRA 4-bit, LoRA r=16, lr=1e-5, 1 epoch
- 15060 records: 9060 Thai (4267 SFT + 265 R1-distill) + 6000 EN (MetaMath/OpenCoder/OpenHermes)
- **Result: catastrophic forgetting — math500 dropped 93.9→54.8 (-39pp)**

### Run 2: Anti-forgetting SFT
- Same base, QLoRA, LoRA r=8 (half), lr=2e-6 (5× lower), 0.3 epoch (3× less)
- 7495 records: 1500 Thai cap + 6000 EN
- **Result: math500 recovered slightly to 66.0 but still -28pp from base; AIME24 unchanged at 23.3**

## Key learnings

1. **R1-Distill is fragile.** ANY supervised fine-tuning damages its math reasoning. Even at LR=2e-6 with tiny LoRA r=8, we lost ~28pp on MATH500.

2. **Our AIME24 eval might undercount.** Both v1 and v2 got identical 7/30 = 23.33% — suspicious. Likely max_tokens=4096 truncates R1's long CoT mid-reasoning for hard AIME problems. R1 sometimes needs 8000+ tokens.

3. **vast.ai interruptible is too unstable for serious work.** 6 preemptions in 11 hours killed multiple eval runs and a training session. Bid raised to $1.20/GPU/hr but spot still contested.

4. **Thai language SFT alone can't reach ThaiExam 75.** Base R1-Distill lacks Thai world knowledge (history, literature). Light SFT on Thai data is insufficient — needs CPT (continued pretraining) on Thai text first.

## What would work (Run 3+)

### For MATH500/AIME24 targets
**Don't SFT R1-Distill at all.** Use base model + inference-time tricks:
- Self-consistency Maj@16-64 (DeepSeek paper: +10-20pp on AIME24)
- DeepConf confidence-weighted Maj@N (+20.9pp on AIME24)
- Python tool-use (SimpleTIR: +28pp on AIME)
- Bumping max_new_tokens to 8192 may already fix our AIME undercount

### For ThaiExam target
Either:
- Start from ThaiLLM-8B (already 48% on ThaiExam) + careful SFT
- Or CPT R1-Distill on Thai text (~10B tokens) then SFT — expensive, needs A100 weeks

### For "single model meeting all targets"
Likely not achievable at 14B without either:
- A 14B base pretrained on math+code+Thai (doesn't exist publicly)
- Multi-stage: CPT → SFT → DPO — needs much more compute
- Mixture-of-experts routing — adds latency

## HF artifacts pushed

- Dataset: `Jnx03/kanitakorn-th-sft-v4-distill` — 4267 SFT + 291 R1-distilled traces
- Adapter v1: `Jnx03/kanitakorn-r1d-qwen14b-stage1-v1-exp-20260521` — aggressive recipe (broken)
- Adapter v2: `Jnx03/kanitakorn-r1d-qwen14b-stage1-v2-20260521` — anti-forgetting recipe (partial recovery)

## Total spend

- Compute: ~$4.00 (vast.ai interruptible 2× A100)
- OpenRouter R1 distill: ~$2 (estimate, hit credit limit during distill)
- Storage: persistent disk fees (~$0.50/day)

## Time spent

~11 hours wall clock from instance provisioning to current state. Effective compute time ~6 hours due to preemptions.

## Recommendation

**Halt this approach.** The fundamental constraint is the choice between:
(a) Math/code reasoning (R1-Distill-Qwen-14B base, 0% Thai)
(b) Thai capability (ThaiLLM-8B, weak math)

No single 14B base today has BOTH. To unify them requires either:
- CPT on a R1-Distill base with 10B+ Thai tokens (weeks of A100 time)
- Or LLM merging via mergekit / TIES / DARE — research-grade work

**Next sensible steps:**
1. Pursue base R1-Distill + Cons@N for math/aime/livecodebench targets only
2. Use ThaiLLM-8B + SFT for ThaiExam/OpenThaiEval/IFEval-TH targets
3. Document as "two specialist models" — honest about the trade-off
