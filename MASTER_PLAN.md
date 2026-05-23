# Master Plan — Kanitakorn 5-Day Campaign
**Synthesized from 5 AI research agents** (ChatGPT Pro x4 + Gemini Pro x1)

## Brutal Honesty (per all agents)
12/12 sweep in 5 days on $80 is **NOT realistic**. Realistic deliverable:
- **8-10 of 12 benchmarks beat** with combined SFT + DPO + merge + inference tricks
- **AIME24 ✅** already beat
- **MATH500** can hit 70-76 with Maj@N + DeepConf (target 82 hard)
- **ThaiExam 67-72** realistic (75 only with breakthrough)
- **IFEval-TH 70-83** achievable via verifier-driven DPO
- **MT-Bench-TH 85+** likely (target within judge noise)

## v8 ALREADY DONE — Eval Running

## v9 RECIPE (Day 2)

### Critical Lesson from Agents
- ThaiLLM official ThaiExam uses **LOGITS-MCQ** not generation (we may be optimizing wrong)
- Every Thai SFT sample MUST include reasoning trace (forced Thai-CoT)
- LoRA r=8/16 too small for multi-objective → use **r=32 or 64**

### v9 Hparams
```yaml
base: deepseek-ai/DeepSeek-R1-Distill-Qwen-14B
method: QLoRA (4-bit NF4 + bf16 compute)
adapter: DoRA if available, else LoRA
lora_r: 32  # was r=8/16 in v6-v8
lora_alpha: 64
dropout: 0.05
target_modules: q_proj k_proj v_proj o_proj gate_proj up_proj down_proj
lr: 1e-4  # higher than before (was 2e-6)
scheduler: cosine, warmup 3%
grad_clip: 1.0
optimizer: paged_adamw_8bit
effective_batch: 64 sequences
seq_len: 4096  # for Thai exam content
epochs: 1.0-1.3
loss: completion-only (mask prompt)
```

### v9 Data Mix (per Gemini + Tab 1)
- 30% NEW: Typhoon-2-70B distilled ThaiExam CoT traces (~$15 API)
- 20% LIMO English math preservation
- 20% Thai SFT validator-passing IFEval positives
- 15% Wangchan filtered (best 8K of 15K)
- 10% Thai-CoT language-lock training data
- 5% Math rehearsal

## v10 DPO (Day 3) — TWO SEPARATE RUNS

### Run 1: Thai-CoT Language Lock
```yaml
method: TRL DPOTrainer + LoRA r=16
beta: 0.1
lr: 1e-6  # very low
epochs: 1.0
label_smoothing: 0.05
loss: sigmoid + 0.02 SFT auxiliary
batch: 64 pairs
pairs: 1000-1500 Thai-CoT vs English-CoT
chosen: Claude REPAIR of model's English CoT into Thai (preserves numbers/formulas)
rejected: current model's actual English/Chinese reasoning
```

### Run 2: IFEval Verifier-Aware DPO
```yaml
pairs: 2500-5000 (target 5K for 55→75-78)
ranking: lexicographic verifier (hard_pass > %_pass > thai_score > task_relevance)
mix:
  25% length failures (no comma, word count)
  20% punctuation failures
  20% language-mixing failures
  15% keyword/count failures
  10% format/list failures
  10% reward-hacking failures
inference: rejection sampling k=4 (return first all-pass)
```

## v11 TIES Merge (Day 4)

### Per Gemini + Tab 4
TIES merge final v10 (post-DPO) with **Qwen2.5-14B-Instruct** to boost IFEval.
Critical: R1-Distill IS Qwen2.5-based — same architecture compatible.

```yaml
merge_method: dare_ties  # or ties
base: deepseek-ai/DeepSeek-R1-Distill-Qwen-14B
models:
  - kanitakorn-v10:
      weight: 0.70  # Thai-heavy weight
      density: 0.50
  - Qwen/Qwen2.5-14B-Instruct:
      weight: 0.30  # IFEval boost
      density: 0.50
lambda: 1.00
dtype: bfloat16
```

## Inference Recipe (Day 4-5)

### DeepConf (test-time, no training)
Filters low-confidence reasoning traces via token entropy. **Free boost on AIME/MATH.**
Integrates with vLLM. Expected: +5-10pp on math.

### Per-Task Configs
| Task | Generation Config |
|------|-------------------|
| IFEval-TH | T=0, top_p=1, no thinking, no <think> leak |
| MT-Bench-TH | T=0.1-0.2, no thinking |
| ThaiExam generated | T=0.6, top_p=0.95, Maj@8-16 |
| **ThaiExam OFFICIAL** | **LOGITS-MCQ** (must use this!) |
| MATH/AIME | Thinking + Maj@16 + DeepConf |

## Key Don'ts
1. ❌ DON'T pivot to Qwen3-14B (arch mismatch, lose 4-day investment)
2. ❌ DON'T train on exact ThaiExam questions (contamination)
3. ❌ DON'T expand tokenizer (Typhoon-2 stopped this)
4. ❌ DON'T use plain Model Soup (use TIES/DARE-TIES)
5. ❌ DON'T do full CPT sub-1B (only +2-4pp ThaiExam, time better spent on SFT/DPO)
6. ❌ DON'T train one massive multi-task LoRA — train SEPARATE adapters then merge

## Critical Bug to Fix
**Language-control**: model produces "Alright,", "好的" for Thai questions.
- Fix: DPO Run 1 (Thai-CoT chosen vs English/Chinese rejected)
- Generate rejected from CURRENT model (real failures), use Claude to REPAIR into Thai chosen

## $30-50 Claude API Budget Allocation
- $15: Typhoon-2-70B distillation via OpenRouter (2K ThaiExam CoT traces)
- $20: Claude Haiku/Sonnet for repair (Thai CoT translations + IFEval chosen)
- $10: MT-Bench LLM-judge for final eval
- $5: Buffer/retry

## Success Criteria
- ✅ **MUST**: AIME24, MT-Bench-TH (already at threshold)
- ✅ **LIKELY**: IFEval-TH (with verifier DPO), Thai improvements over base
- ⚠️ **STRETCH**: MATH500 82+, ThaiExam 75+
- ❌ **HARD**: OpenThaiEval 80, LiveCodeBench-TH 35, HotpotQA 46 (untested)

## Sources
1. ChatGPT Pro Tab 1: 4-day strategic plan + v8/v9/v10 hparams
2. ChatGPT Pro Tab 2: CPT analysis (300-400M tokens → +1.5-4pp realistic)
3. ChatGPT Pro Tab 3: DPO recipe (two separate runs)
4. ChatGPT Pro Tab 4: LoRA merging (TIES > soup)
5. Gemini Pro: SOTA techniques + DeepConf + OpenThaiGPT recipe
