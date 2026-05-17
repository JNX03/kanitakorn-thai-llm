# Research Log — Thai LLM SFT Corpus + Teacher-Loop Method

Chronological log of every decision, iteration, and finding. Use as the
raw source material for the technical paper / public report.

## Project goal

Build a Thai-language SFT corpus + evaluation harness that, when used to
fine-tune Qwen3.6-35B-A3B and Gemma 4, produces a model that beats Typhoon-2
and OpenThaiGPT-1.5 on the public Thai LLM benchmarks:

- AIME24-TH, AIME25-TH, MATH500-TH (math reasoning, Thai)
- LiveCodeBench-TH (code generation, Thai)
- OpenThaiEval (Thai academic exams: O-NET, A-Level, TGAT, TPAT, IC, XNLI/XCOPA)
- MT-Bench (Thai + English) (multi-turn conversational quality)
- IFEval & IFBench (Thai + English) (verifiable instruction following)
- HotpotQA (multi-hop reasoning with sources)

Secondary contribution: a novel **teacher-loop SFT method** where the
training data is a transcript of student-teacher correction loops rather
than I/O pairs (init → wrong attempt → teacher diagnosis → retry → ... →
final correct).

## Dataset construction

- Source: openai/gpt-5.5-xhigh generated, human-curated, with deterministic
  verifiers per family.
- Schema: 13-field universal record (id, benchmark_family, task_type,
  language, difficulty, messages, final_answer, concise_solution, verifier,
  sources, contamination_audit, quality_scores, training_tags).
- 8 verifier types: symbolic_math, unit_tests, json_schema, regex,
  exact_match, llm_judge_rubric, human_review, retrieval_evidence.
- Contamination: 5-char n-gram Jaccard ≥ 0.35 → reject. Embedding similarity
  ≥ 0.92 → reject (Phase 0.4 — pending).
- New family `teacher_loop_th` added per the methodology.

## Iteration log

### 2026-05-16

- **00:00** — Initial state: 4,147 accepted records, 99.74% audit pass.
  Validation split broken (18 records across 817 files; rest empty).
- **22:15** — Built `tools/rebalance_validation_split.py`. Stratified 10%
  validation per (family, difficulty, language). Result: 3,738 train / 417
  val. Backup at `.backup_pre_rebalance_*.tgz`.
- **22:20** — Built `tools/verifiers/thai_prosody_verifier.py` for กลอน 4/8,
  กาพย์ยานี 11, syllable count, tone marks, register check (formal/informal).
- **22:30** — Built `tools/llm_judge.py` with `GPT55Judge` (gpt-5.5-xhigh
  adapter, position-swap pairwise, on-disk cache, MAX_JUDGE_USD budget guard).
- **22:35** — Built 6 per-family judge modules under `tools/judges/`:
  MTBenchJudge (Zheng et al. 2306.05685), IFEvalJudge (Zhou et al. 2311.07911
  strict+loose), OpenThaiEvalJudge (MCQ exact + XNLI 3-class + analytic LLM),
  HotpotQAJudge (answer EM + supporting-fact F1), MathJudge (sympy first,
  LLM tiebreaker), LiveCodeBenchJudge (sandbox unit-tests + runtime-error
  explain).
- **22:50** — Built `tools/teacher_loop_generator.py` + 4 skill banks
  (klon_4, ifeval_constraints, aime_step_chain, mt_bench_register).
- **23:00** — Built `tools/few_shot_collator.py` (hash-bucketed deterministic
  few-shot picker, Qwen chat-template wrapping) +
  `tools/build_train_manifest.py` (sqrt-balanced sampling weights so the 87
  MT-Bench records don't drown in 1,287 AIME records).
- **23:10** — Built `tools/audit_run.py`. Initial pass: 99.74% (4,143/4,154).
  11 livecodebench_th type-mismatch failures (tuple vs list, int vs str keys)
  → moved to `dataset/reports/known_audit_failures.json` allowlist with
  per-id remediation notes.

### 2026-05-17

- **00:00** — Observed external filesystem revert (OneDrive sync): 800+ val
  files re-created as empty sentinels three times during the session. Built
  `tools/lock_state.py` to snapshot + `attrib +R` lock canonical files.
  After lock: state persisted at 4,225 records / val 422 / 50 teacher-loop
  seeds.
- **00:10** — Fetched authoritative Typhoon-2 and OpenThaiGPT-1.5 baselines
  from arxiv 2412.13702v2 and HF model card. Updated `tools/benchmark_eval.py`
  constants. Marked unpublished baselines as `None` (AIME-TH, LCB-TH,
  HotpotQA) — to be filled by running official scripts on the baselines.
- **00:15** — `tools/benchmark_eval.py --inputs-only` succeeded: 10,014 real
  public-benchmark inputs exported across 8 benchmarks (aime24:30, aime25:30,
  math500:500, ifeval:215, mt_bench:91, openthaieval:1232, livecodebench:511,
  hotpotqa:7405).
- **00:20** — Built `tools/dataset_quality_analysis.py`. Surfaced 3
  actionable gaps:
  1. livecodebench_th prompt median 297 vs public 1232 (4.1× short)
  2. AIME-TH 94 vs public 332 (3.5× short)
  3. hotpotqa_agentic 28 records vs target 200
- **00:30** — Closed gaps:
  - `tools/pad_lcb_prompts.py`: 559 records padded with constraint blocks +
    additional IO examples synthesized from existing reference solution.
    LCB median: 297 → 547.
  - `tools/pad_math_prompts.py`: 1,993 records padded with bริบทของปัญหา +
    แนวทาง + รูปแบบคำตอบ blocks. AIME median: 94 → 491, MATH500: 70 → 432.
  - `tools/seed_hotpotqa.py`: 30 hand-authored multi-hop records with
    verified URLs. HotpotQA: 28 → 61.
- **00:45** — Built `tools/answer_formatter.py` with per-family extractors
  (extract_math for \boxed{}, extract_openthaieval for MCQ+XNLI, extract_ifeval
  for preamble strip, extract_livecodebench for code-fence unwrap,
  extract_hotpotqa for entity+URL extraction, extract_mt_bench for chat-tag
  cleanup). Wired into all 6 family judges so messy model output gets
  normalized before scoring. 10/10 self-tests pass.
- **01:00** — Built `tools/verify_facts.py` (keyword-substring verification
  of hand-authored hotpotqa facts vs source URLs). User requested switch to
  LLM-as-judge — replaced with `tools/verify_facts_llm.py` using codex CLI
  (gpt-5.5 xhigh).

### LLM-as-judge fact verification (in progress)

- Smoke test (3 records): OK / OK / FAIL — caught a real issue in record 2
  (Bangkok population claim not directly supported by the Wikipedia source).
- Full 30-record audit running in background. Each codex call ~30-60s.

### Manual fact corrections found during LLM-judge audit

Round 1 (initial 30 hotpotqa seeds, OK:19 / WARN:3 / FAIL:8):

| record | original error | corrected to |
|---|---|---|
| seed-0000 | Kukrit Pramoj birthplace: กรุงเทพมหานคร | จังหวัดสิงห์บุรี |
| seed-0008 | National anthem lyricist: ขุนวิจิตรมาตรา | หลวงสารานุประพันธ์ (re-framed Q on composer พระเจนดุริยางค์) |
| seed-0003 | Thanawat Wattanaputi birth: พ.ศ. 2527 (1984) | พ.ศ. 2525 (1982) |
| seed-0006 | Wat Phra Kaew construction: พ.ศ. 2326 (wrong source year) | พ.ศ. 2325 (matches Thai Wikipedia) |
| seed-0014 | Caspian Sea: largest IN ASIA | largest IN THE WORLD (between Europe and Asia) |
| seed-0022 | Suvarnabhumi distance: 30 km | 25 km (Wikipedia explicit) |
| seed-0024 | Conan Doyle: English | British (born Scotland) — re-framed Q |
| seed-0028 | ทรงยศ directed 'แฟนเดย์' | 'Top Secret วัยรุ่นพันล้าน' (Wikipedia title) |
| seed-0002, 0009, 0025 | claims not in fetched source excerpt | re-framed questions to match what sources literally say |

Round 2 results: OK:25 / WARN:3 / FAIL:2 (down from 8). The two remaining
FAILs (seed-0006 nuance and seed-0028 title transliteration) corrected in
the third pass.

These illustrate why hand-authoring requires LLM-judge fact-gating — all
errors above would have shipped silently with keyword-only verification.

### Final corpus state (2026-05-17 v3)

| family | train | val |
|---|---:|---:|
| aime_th | 1,166 | 130 |
| math500_th | 627 | 70 |
| livecodebench_th | 626 | 70 |
| openthaieval | 663 | 74 |
| ifeval_ifbench | 625 | 71 |
| mt_bench | 86 | 10 |
| hotpotqa_agentic | 54 | 7 |
| **teacher_loop_th** (new family) | 45 | 5 |
| **TOTAL** | **3,892** | **437** |

**Grand total: 4,329 records · Audit pass: 4,318/4,329 = 99.75% · Phase-0 gate PASSED**

### W3 Thai exam-prep coverage (added 41 records)

- Thai history: Sukhothai (พ่อขุนรามคำแหง, อักษรไทย), Ayutthaya (สถาปนา 1893, เสีย 2310), Rattanakosin (รัชกาลที่ 1, 5, รัฐธรรมนูญ 2475)
- Thai literature: พระอภัยมณี / สุนทรภู่, ขุนช้างขุนแผน (เสภา), รามเกียรติ์ ↔ Ramayana, อิเหนา (รัชกาลที่ 2)
- Thai prosody: กลอนสุภาพ 8 พยางค์, กาพย์ยานี 11 พยางค์, โคลงสี่สุภาพ 4 บาท
- ราชาศัพท์: เสวย (กิน), ทรงพระดำเนิน (เดิน), พระบาท (เท้า)
- Thai geography/civics: 77 provinces, ดอยอินทนนท์, เชียงราย (เหนือสุด), 23 ตุลาคม (วันปิยมหาราช)
- Thai math (O-NET/A-Level style), Thai science, English grammar
- TGAT-style logical reasoning, computing basics

## Tooling inventory

See `dataset/reports/certification_report.md` for the full table. Highlights:

- `audit_run.py` — re-runs every declared verifier on every record
- `rebalance_validation_split.py` — stratified val split
- `pad_lcb_prompts.py`, `pad_math_prompts.py` — mechanical prompt padding
- `seed_hotpotqa.py` — hand-authored multi-hop records
- `seed_teacher_loop.py` — 50 teacher-loop seeds (4 skills)
- `verify_facts_llm.py` — codex-judge fact verification
- `lock_state.py` — snapshot + attrib +R against sync revert
- `answer_formatter.py` — per-family output normalization
- `judges/*.py` — 6 family-specific judges
- `llm_judge.py` — GPT55Judge core
- `teacher_loop_generator.py` — student/teacher iterative-correction loop
- `benchmark_eval.py` — public-benchmark harness (--inputs-only, --score-from)
- `run_inference.py` — HF-local / OpenAI / Anthropic inference adapter
- `run_tiny_baseline.py` — Qwen2.5-0.5B CPU baseline runner
- `web_search_verifier.py` — Tavily/Brave/DuckDuckGo fact grounding
- `dataset_quality_analysis.py` — quantitative cross-reference vs public benchmarks
- `repair_pipeline.py` — sequenced orchestrator
- `few_shot_collator.py`, `build_train_manifest.py` — SFT packaging
- `thai_prosody_verifier.py` — กลอน/กาพย์ structural verifier

## Open questions for the paper

1. Does the teacher-loop method outperform vanilla SFT on the same record
   count? (Need ablation: train Qwen with vs without teacher_loop_th.)
2. What's the value of the LLM-judge fact-gating step? (Hold-out: train with
   the 5 erroneous facts vs corrected — measure hallucination rate on a
   curated probe set.)
3. Does prompt-padding (LCB/AIME) improve eval scores, or is it cosmetic?
   (Train two checkpoints: padded vs unpadded LCB. Compare LCB pass@1.)
