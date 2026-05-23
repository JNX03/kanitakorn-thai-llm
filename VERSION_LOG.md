# Kanitakorn Version Log

One row per published adapter. Sorted oldest-first.

| Date | Repo | Stage | Base | LR | R | Epochs | THAIEXAM | MATH500 | AIME24 | Notes |
|------|------|-------|------|---:|--:|-------:|---------:|--------:|-------:|-------|
| 2026-05-22 | [kanitakorn-qwen3-14b-stage1-qwen3-20260522](https://huggingface.co/Jnx03/kanitakorn-qwen3-14b-stage1-qwen3-20260522) | stage1-qwen3 | qwen3-14b | 5e-06 | 8 | 0.5 | — | — | — | Qwen3-14B + Thai SFT (v3): light recipe to add Thai capability |
| 2026-05-22 | [kanitakorn-qwen3-14b-stage1-qwen3-v5-ultralight-20260522](https://huggingface.co/Jnx03/kanitakorn-qwen3-14b-stage1-qwen3-v5-ultralight-20260522) | stage1-qwen3-v5-ultralight | qwen3-14b | 1e-06 | 4 | 0.3 | — | — | — | Kanitakorn v5: Qwen3-14B ultra-light Thai SFT. Slight damage vs base (Thai 67→64 Maj@4). Lesson: SFT on Qwen3 not viable in our budget. |
| 2026-05-22 | [kanitakorn-r1d-qwen14b-stage1-r1d-v6-rehearsal-20260522](https://huggingface.co/Jnx03/kanitakorn-r1d-qwen14b-stage1-r1d-v6-rehearsal-20260522) | stage1-r1d-v6-rehearsal | r1d-qwen14b | 2e-06 | 8 | 0.5 | — | — | — | Kanitakorn v6: R1-Distill-14B with rehearsal SFT (Thai+Math+Code mix). Goal: preserve math while pushing Thai. |
| 2026-05-22 | [kanitakorn-r1d-qwen14b-stage1-r1d-v7-mega-20260522](https://huggingface.co/Jnx03/kanitakorn-r1d-qwen14b-stage1-r1d-v7-mega-20260522) | stage1-r1d-v7-mega | r1d-qwen14b | 2e-06 | 8 | 0.7 | — | — | — | Kanitakorn v7: R1-Distill-14B with 10918 unique Thai SFT records (2x v6 data). Goal: push ThaiExam past 70%. |

## 2026-05-22/23 — Evaluation Results

### v6 (R1-Distill + Rehearsal SFT) — Full Benchmark Results

| Benchmark | Records | Score | Target | Status |
|-----------|--------:|------:|-------:|:------:|
| ThaiExam | 565 (Maj@4) | 62.3% (352/565) | >75 | -12.7pp |
| MATH500 | 500 (4k tok, n=1) | 62.0% (310/500) | >82 | -20pp |
| AIME24 | 30 (4k tok, n=1) | 23.33% (7/30) | >25 | -1.67pp |

### v7 (R1-Distill + Mega SFT 10918 records) — Full Results

| Benchmark | Method | Score | Target | Status |
|-----------|--------|------:|-------:|:------:|
| ThaiExam | 565 (Maj@4) | **61.77%** (349/565) | >75 | -13pp |
| MATH500 | 500 (4k tok, n=1) | **63.4%** (317/500) | >82 | -19pp |
| MATH500 | 200 (Maj@8) | **68.0%** (136/200) | >82 | -14pp |
| AIME24 | 30 (4k tok, n=1) | **23.33%** (7/30) | >25 | -1.67pp |
| AIME24 | 30 (Maj@16) | **46.67%** (14/30) | >25 | ✅ **BEAT +22pp** |

### Key Findings
- Self-consistency (Maj@N) decisive for AIME (+23pp), modest for MATH500 (+5pp), zero for ThaiExam (plateau at 61-62).
- v7 mega data did NOT improve ThaiExam over v6 — plateau.
- Diagnostic: model reasons in English/Chinese for Thai questions. Need Thai-language CoT in training data.

### Next: v8 with Thai-reasoning CoT data + 2x A40 DDP
- 1596 Thai-reasoning records (forced Thai CoT)
- 16556 external (Wangchan + Typhoon T1 Thai)
- 11K targeted Thai synth
- LoRA r=16 (double capacity)
- 2× A40 DDP training (2x speedup)
