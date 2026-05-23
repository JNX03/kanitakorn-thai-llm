# Multi-Agent Research Insights (2026-05-23)

5 parallel research agents (ChatGPT Pro x4 + Gemini Pro x1) consulted.

## Critical Findings

### 1. Optimization Target Mismatch (DEEP BUG)
**ThaiLLM official ThaiExam uses LOGITS-MCQ scoring, not generation.** We've been optimizing Maj@4 generation with rationales. Need to:
- Train answer-only MCQ format too
- Eval in BOTH logits mode AND generation mode
- 565 questions across 5 splits (ONET 167, IC 100, TGAT 70, TPAT-1 121, A-Level 132)

### 2. DeepConf — Test-Time Free Boost
- Filters low-confidence reasoning traces using localized token entropy
- Zero training required, integrates into vLLM
- Expected: MATH/AIME score boost without extra training cost

### 3. Typhoon-2-70B Distillation (~$10-15)
- Generate 2000 ThaiExam-style CoT traces via API
- Every sample MUST have reasoning trace (we may be missing this)
- Mix with 1000 English LIMO examples for math preservation

### 4. OpenThaiGPT-1.6 Recipe Replication
**Task arithmetic with weighted LoRA merge:**
- General: 0.15
- Translation: 0.15  
- Thai Exams: **0.70** (heavy weight)

### 5. TIES Merge with Qwen2.5-14B-Instruct
- R1-Distill-Qwen-14B IS Qwen2.5-based (same architecture)
- TIES merge for IFEval-TH boost (could close -27pp gap)
- DON'T merge with Qwen3-14B (architectural mismatch)

### 6. LoRA Capacity Under-sized
- v6/v7 used r=8 — TOO SMALL for multi-objective adapter
- Use **r=32 DoRA** for v8/v9
- Alpha 64 for r=32

### 7. Don't Pivot to Qwen3
- R1-Distill has reasoning baked in
- Switching loses 4-day investment
- Stay with R1-Distill-Qwen-14B

### 8. Validator-in-Loop IFEval Training
- Only train on samples that PASS the actual evaluator
- DPO with chosen=passing vs rejected=one-violation
- Expected: IFEval-TH 55 → 75-83

## Honest Expected Outcomes (Per Research)

| Benchmark | Target | Current | Realistic |
|-----------|-------:|--------:|----------:|
| AIME24 | >25 | 47 (Maj@16) | ✅ already beat |
| MATH500 | >82 | 68 (Maj@8) | 70-76 (low confidence to hit 82) |
| ThaiExam | >75 | 62 (Maj@4) | 67-72 (75 only if breakthrough) |
| IFEval-TH | >82 | 55 | 75-83 (achievable with verifier-in-loop) |
| MT-Bench-TH | >85 | 84.5 | within judge noise — preserve |

## 4-Day Plan (Synthesized from agents)

### Day 1 (TODAY): v8 SFT (RUNNING — almost done)
- R1-Distill base + r=16 LoRA + 11K Thai SFT + math rehearsal
- ~10 min remaining

### Day 2: Distill from Typhoon-2-70B + v9 SFT
- $10-15 to OpenRouter for 2000 ThaiExam CoT traces
- Build v9 with r=32 DoRA, LR 1e-4, 1 epoch
- Both rationale + answer-only formats
- 60% Thai exam targeted, 25% IFEval validator-positive, 15% math preserve

### Day 3: DPO Language-Lock + v10 Constraint Repair
- DPO chosen=Thai-CoT vs rejected=English-CoT (fix English-reasoning bug)
- DPO chosen=IFEval-pass vs rejected=one-violation
- LR 5e-6, beta 0.05, 2-3K pairs

### Day 4: TIES Merge + DeepConf Inference + Final Eval
- TIES merge final adapter with Qwen2.5-14B-Instruct for IFEval boost
- DeepConf for AIME/MATH inference
- Logits-MCQ eval mode for ThaiExam
- Push final model to HF

## Source Agents
- ChatGPT Pro Tab 1: Strategic 4-day plan
- ChatGPT Pro Tab 2: CPT analysis (still generating)
- ChatGPT Pro Tab 3: DPO recipe (still generating)
- ChatGPT Pro Tab 4: LoRA merging (still generating)
- Gemini Pro: SOTA techniques + honest assessment
