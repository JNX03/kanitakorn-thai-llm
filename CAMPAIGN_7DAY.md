# 7-Day Campaign — Plan & Status

## Targets (must beat ALL)

| Benchmark | Target | Current ours | Peer best | Δ to target |
|-----------|-------:|-------------:|----------:|------------:|
| AIME24-TH | >15 | n/a | OpenThaiGPT-1.6-72B: 6.67 | +15 |
| AIME24 | >25 | n/a | OpenThaiGPT-1.6-72B: 23.33 | +25 |
| MATH500-TH | >56 | n/a | OpenThaiGPT-1.6-72B: 43.2 | +56 |
| MATH500 | >82 | 0.540 (judge) | OpenThaiGPT-1.6-72B: 82.0 | +28pp |
| LiveCodeBench-TH | >35 | n/a | OpenThaiGPT-1.6-72B: 32.43 | +35 |
| LiveCodeBench | >60 | n/a | OpenThaiGPT-1.6-72B: 54.21 | +60 |
| OpenThaiEval | >80 | n/a | OpenThaiGPT-1.6-72B: 78.7, Typhoon-S: 67.1 | +80 |
| HotpotQA (TH+EN) | >46 | n/a | Typhoon-S TH: 37 | +46 |
| Instruction-following (TH+EN) | >57 | n/a | — | +57 |
| MT-BENCH (TH+EN) | >85 | n/a | Typhoon-S TH: 78.9 | +85 |
| THAIEXAM | >75 | 0.413 (judge) | OpenThaiGPT-1.5: 0.520, Pathumma: 0.513 | +34pp |
| IFEval TH | >82 | n/a | Typhoon-S: 76.45 | +82 |

**These targets are very aggressive — many above OpenThaiGPT-1.6-72B's published scores.**
Hitting them on an 8B model requires: best base + comprehensive corpus + multi-stage SFT + inference-time tricks (self-consistency, possibly ensemble).

## Strategy

### Stage 1: Foundation SFT (Days 1-2)
Train Qwen3-8B-Base (or ThaiLLM-8B) on combined corpus:
- 4,267 SFT-ready records (manifest-weighted) covering all 8 benchmark families
- Mixed in: MetaMathQA (~5k) + OpenCoder evol_instruct (~5k) + OpenHermes-2.5 (~5k)
- LoRA r=32 (more capacity than r=16), 3 epochs, save_steps=200 for preemption recovery
- 2× A100 DDP via `torchrun --nproc_per_node=2`

### Stage 2: Specialist SFT rounds (Days 3-5)
After Stage 1 baseline eval, train targeted specialists:
- **Math specialist**: extend with NuminaMath + AIME problem set, focus on chain-of-thought
- **Code specialist**: extend with HumanEval-style problems + OpenCoder
- **Thai exam specialist**: focus on OpenThaiEval + ThaiExam with high replication
- Each on its own LoRA adapter (separable for merging)

### Stage 3: Task-arithmetic merge sweep (Day 6)
Combine Stage 1 + specialists via weighted-sum:
- α/β/γ sweep across {0.5/0.25/0.25, 0.6/0.2/0.2, 0.7/0.15/0.15, ...}
- Pick best by aggregate target-margin

### Stage 4: Self-consistency inference (Day 7)
For final eval: n=5 majority vote on math/MCQ benchmarks. Typical +5-10pp.

## Files prepared (this prep session)

| File | Purpose |
|------|---------|
| `train_2xa100_interruptible.py` | Multi-GPU DDP SFT with auto-resume, manifest-weighted, EN mix |
| `fire_2xa100.sh` | Vast.ai launch script — torch pin, deps, tmux loop with auto-retry |
| `eval_self_consistency.py` | n-sample majority vote eval for AIME/MATH/ThaiExam |
| `CAMPAIGN_7DAY.md` | This doc |

Existing infrastructure (from prior session, already in place):
- `dataset/sft_ready/*_train.jsonl` — 4267 records across 8 families, pre-formatted
- `dataset/sft_ready/manifest.json` — weighted sampling spec
- `tools/llm_judge.py`, `tools/judges/*.py` — per-family LLM-as-judge
- `tools/benchmark_eval.py` — public benchmark harness
- `tools/repair_pipeline.py` — dataset audit/repair orchestrator

## Launch sequence (when 2× A100 instance is ready)

```bash
# Local — sync code (skip large dirs):
rsync -avz -e "ssh -p <PORT>" \
    --exclude='runs/' --exclude='__pycache__/' --exclude='*.log' \
    ./ root@<HOST>:/workspace/kanitakorn/

# Remote — fire:
ssh -p <PORT> root@<HOST> 'cd /workspace/kanitakorn && \
    HF_TOKEN=hf_... OPENROUTER_API_KEY=sk-or-... \
    SFT_BASE=Qwen/Qwen3-8B-Base SFT_OUT=/workspace/kanitakorn/runs/stage1_v1 \
    SFT_EPOCHS=3 SFT_LORA_R=32 SFT_MIX_EN=1 \
    bash fire_2xa100.sh'

# Monitor:
ssh -p <PORT> root@<HOST> 'tail -F /tmp/train_campaign.log'
```

## Preemption handling

The `fire_2xa100.sh` script wraps training in a `while true` loop. On preemption:
1. Vast.ai pauses the container, you may need to manually resume in the UI
2. After resume, SSH back in and run `bash fire_2xa100.sh` again
3. The training auto-detects the latest checkpoint and resumes
4. Checkpoint cadence is 200 steps (~20 min on 2× A100) — max lost work is ~20 min

## Model choice — A/B TEST EMPIRICALLY

User directive: don't pick a base in advance — test on the GPU server and pick the winner.

### Day 0 (first 4 hours after A100 lands): bake-off
Train each candidate on a small subset (500 records, 1 epoch, ~30 min each) and eval on a tight subset of every target benchmark (100-record samples). Tournament:

1. **Qwen3-8B-Base** — strong math/code, weaker Thai (~0.31 ThaiExam)
2. **ThaiLLM/ThaiLLM-8B** — +17pp Thai pre-SFT but weaker math base
3. **Qwen3-14B-Base** — bigger headroom, tight on 2× 40GB (LoRA r=16, batch 1)
4. **Qwen2.5-7B-Instruct** — already instruct-tuned, may need just light specialization

Pick winner by aggregate Δ-to-target across all 12 benchmarks. Use `ab_base_compare.py` (next).

### Corpus mix — A/B TEST EMPIRICALLY
- Mix A: Typhoon-2 30:70 (10k EN/math/code on top of 4267 TH)
- Mix B: 50:50 (5k EN/math/code)
- Mix C: Pure Thai (control, expect regression)

Run 3 short epochs with each mix on the winning base, evaluate. Adopt the winner for full Stage 1.

## Stretch plays if behind target on Day 5

1. **Distillation from Qwen3-72B**: generate CoT solutions on AIME/MATH problems with the 72B (via OpenRouter), train on (prompt → 72B-CoT) pairs. Big lift for math.
2. **Best-of-N**: at eval time, sample 16 responses, use small Qwen reward model to pick best. Gets +10-15pp on AIME at inference cost.
3. **Tool-use for AIME/MATH**: train on records that call a Python interpreter for arithmetic. AIME answers are integers 0-999 → trivial validator. Big lift if base model can be taught to emit `<tool>...</tool>` blocks.
4. **DPO from judge**: after Stage 1, generate paired responses, use gemini-judge to label preferred, DPO-train. Improves MT-Bench and IFEval.

## Honest reality check (PRE-RESEARCH)

Even with everything above, 8B-class models historically peak around:
- ThaiExam: 55-60 (vs target 75)
- MATH500: 70-75 (vs target 82)
- AIME24: 15-25 (vs target 25)
- OpenThaiEval: 65-70 (vs target 80)

## NEW STRATEGY (post-research 2026-05-21)

Research (DeepSeek-R1 paper, Qwen blogs, DeepConf paper) reframes everything:

- **DeepSeek-R1-Distill-Qwen-14B already scores MATH500=93.9, AIME24=69.7, AIME24-Cons@64=80.0** out of the box.
- The hard target is **adding Thai capability without destroying the English math reasoning**.

### Anti-forgetting recipe (the critical insight)

R1-Distill bakes long CoT into the weights. Naive Thai SFT will erase it. Mitigations being implemented:

1. **LoRA r=8-16 only** (small enough to preserve base weights)
2. **Learning rate 5e-6 to 1e-5** (lower than usual)
3. **Mix R1-distilled Thai CoT** (running now via OpenRouter — Thai problems → R1 traces). This teaches Thai+math simultaneously.
4. **Typhoon-2 30:70 Thai:EN ratio** (the EN side keeps R1's reasoning warm)
5. **Monitor MATH500 after every epoch** — if regression > 3pp, stop & lower LR
6. **Eval before training** — establish R1-Distill-14B base scores on top-3. Use as floor.

### R1 distillation in progress

- AIME-TH: 33/200 distilled traces (deepseek-r1 via OpenRouter)
- MATH500-TH: 71/150 distilled traces
- Output format: `{messages: [{user: thai_problem}, {assistant: thai_R1_CoT_with_\\boxed}]}`
- Quality verified — boxed answers, Thai CoT throughout

These get mixed into the SFT corpus at ~5-10% weight. The model sees: "Thai problem in → Thai R1 CoT out" with the same reasoning patterns. This is the key.

### Inference-time amplifiers (Stage D)

| Method | Source | Expected gain on AIME24 |
|--------|--------|------------------------:|
| Self-consistency Maj@64 | DeepSeek-R1 paper | +10-15pp |
| DeepConf (confidence-weighted Maj@64) | arXiv 2025 | +20.9pp (over baseline) |
| SimpleTIR (Python tool-use) | arXiv 2509.02479 | +28pp |
| Cons@64 on R1-Distill-14B | DeepSeek-R1 | 80.0 vs 69.7 single |

Even if SFT doesn't help (or hurts), AIME24 at Cons@64 = 80.0 still smashes target 25.

## Updated reality check

| Target | R1-Distill-14B base | Floor if math preserved | Target |
|--------|--------------------:|------------------------:|-------:|
| AIME24 | ~69.7 | 80.0 (Cons@64) | 25 ✅ |
| MATH500 | ~93.9 | 93.9 | 82 ✅ |
| ThaiExam | unknown — likely 30-45 | needs Thai SFT | 75 ⚠️ |
| OpenThaiEval | unknown — likely 40-55 | needs Thai SFT | 80 ⚠️ |
| IFEval-TH | unknown — likely 50-60 | needs Thai SFT | 82 ⚠️ |
| MT-BENCH-TH | unknown | needs Thai SFT + DPO | 85 ⚠️ |

**Verdict:** the 4 Thai benchmarks are now the bottleneck. The recipe must add Thai without harming the rest. R1-distilled Thai CoT is the key — it teaches Thai with the same reasoning patterns.
