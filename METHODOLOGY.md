# Methodology

## 1. Problem statement

Thai-language large language models (Typhoon-2, OpenThaiGPT-1.5, OpenJAI-v1)
publish strong scores on Thai academic exams (ThaiExam, M3Exam,
OpenThaiEval) but show systematic gaps on (a) math reasoning at AIME / MATH
difficulty, (b) verifiable instruction following, (c) multi-hop reasoning
with citation, and (d) Thai cultural / prosodic specificity (กลอน, ฉันท์,
formal register). This work targets those gaps with a verified SFT corpus
and a novel skill-loop training method.

## 2. Dataset construction

### 2.1 Universal record schema

Every record carries 13 required fields enforced by
`tools/package_and_verify.py::SCHEMA` (JSON Schema Draft 2020-12):

    id, benchmark_family, task_type, language, difficulty, messages,
    final_answer, concise_solution, verifier, sources,
    contamination_audit, quality_scores, training_tags

`teacher_loop_th` records additionally carry `loop_metadata: {n_corrections,
skill_taught, convergence_reached, base_family}`.

### 2.2 Verifier taxonomy

8 deterministic verifier types — each accepted record must pass its declared
verifier in `tools/audit_run.py`:

| type | check | used by |
|---|---|---|
| symbolic_math | `python_asserts` evaluated via sympy | aime_th, math500_th |
| unit_tests | `reference_solution` executed against `public_tests` + `hidden_tests` + `randomized_test_code` | livecodebench_th |
| json_schema | response parses as JSON and validates against given schema | ifeval_ifbench |
| regex | response matches per-task python_asserts (line count, prefixes, keyword frequency, etc.) | ifeval_ifbench |
| exact_match | `accepted_answers` set match | openthaieval |
| llm_judge_rubric | ≥3 criteria + ≥4 messages; with `--use-llm-judge`, per-criterion pointwise score ≥ 8.0 | mt_bench, teacher_loop_th |
| retrieval_evidence | ≥2 sources, ≥2 supporting facts; URLs reachable | hotpotqa_agentic |
| human_review | no-op placeholder | rarely used |

### 2.3 Contamination control

- Normalize text (lowercase, strip whitespace).
- 5-char n-gram Jaccard ≥ 0.35 against cached benchmark inputs → reject.
- If n-gram ≥ 0.25, run simhash similarity; ≥ 0.75 → reject.
- Embedding similarity (planned): `paraphrase-multilingual-MiniLM`,
  cosine ≥ 0.92 → reject.
- Only **hashes** of benchmark text stored locally (manifest at
  `dataset/reports/benchmark_blacklist_hash_manifest.json`); benchmark
  source text never enters the training corpus.

### 2.4 LLM-as-judge fact-gating

For any hand-authored record (especially `hotpotqa_agentic`), the
`tools/verify_facts_llm.py` gate fetches every cited URL, gives the page
text + claimed facts to gpt-5.5-xhigh (via codex CLI), and asks "do the
sources actually support each claim?" Records receiving `FAIL` are
quarantined and not added to the corpus until the author fixes the claim
or replaces the source.

Quantitative impact: of 30 initial hand-authored hotpotqa records, the
LLM-judge audit caught 5 factual errors that keyword-matching alone missed:

- Kukrit Pramoj birthplace (Bangkok → Sing Buri)
- National anthem lyricist (ขุนวิจิตรมาตรา → หลวงสารานุประพันธ์)
- Wat Phra Kaew construction year (2325 → 2326)
- Caspian Sea scope (largest in Asia → largest in world)
- ทรงยศ filmography (แฟนเดย์ → เดอะ บิลเลียนแนร์)

This translates to a ~16.7% error rate on hand-authored content without the
gate. Without LLM-judge verification, these would all have shipped silently.

## 3. Training method

### 3.1 Standard SFT (control)

Per-family SFT bundles assembled by `tools/few_shot_collator.py` with a
deterministic hash-bucketed few-shot picker (3 examples for math families,
2 for code, 1 for mt_bench, 0 for teacher_loop_th since the records are
already lessons). Output written as Qwen-style chat-template text.

Per-family sampling weights are sqrt-balanced
(`tools/build_train_manifest.py`) so the 87 MT-Bench records aren't drowned
by the 1,291 AIME records.

### 3.2 Teacher-loop SFT (novel contribution)

For each skill (กลอน 4 composition, IFEval constraint following, AIME
step-chain, MT-Bench register calibration), the
`tools/teacher_loop_generator.py` runs:

    1. user_init (e.g., "แต่งกลอน 4 เกี่ยวกับฤดูฝน")
    2. student_attempt_1 — emitted by gpt-5.5-xhigh @ temperature 1.0
       with "deliberately learning" system prompt
    3. Run the deterministic verifier on attempt_1
    4. If pass → drop the record (no learning signal)
    5. If fail → teacher_correction — emitted by gpt-5.5-xhigh @
       temperature 0.0 given the verifier's failure trace; the teacher
       NAMES the failure mode + gives a specific fix, never reveals the answer
    6. student_retry → verifier → loop up to N=5 rounds
    7. Final assistant turn must pass the verifier; else quarantine record

Records carry `loop_metadata.n_corrections` (must be ≥ 1 to be kept) and
`convergence_reached` (must be true).

Quality guard: ≥30% of accepted records must have n_corrections ≥ 2
(prevents degenerate loops where the student gets it right too easily).

Schema-validated 50 hand-authored seeds (12 klon_4, 10 register, 16 ifeval,
12 math) live at `dataset/train/train_teacher_loop_th_000.jsonl`. The
`teacher_loop_generator.py` script scales this to 500+ given OPENAI_API_KEY.

### 3.3 Why teacher-loop?

Standard SFT shows the model only the correct output. The teacher-loop
shows the model:
- What incorrect outputs look like for this skill
- What the verifier complains about ("วรรคที่ 1 มี 14 พยางค์ แต่กลอน 4 ต้องการ 4")
- How the teacher diagnoses and corrects
- The successful retry that follows the correction

The hypothesis is that this teaches the model a meta-skill of
*self-correction*, which standard SFT does not — and self-correction at
inference time is what bridges the gap between "can almost do it" and "does
it reliably." Ablation (planned): train two checkpoints from the same
base, one with vs one without teacher_loop_th, measure delta on hard splits
(AIME hard, IFEval strict, ฉันทลักษณ์ generation).

## 4. Evaluation protocol

### 4.1 Public benchmarks

`tools/benchmark_eval.py --inputs-only` exports the public input files
(10,014 records across 8 benchmarks). The trained model emits predictions
via `tools/run_inference.py --backend hf-local/openai/anthropic`.
`tools/benchmark_eval.py --score-from <preds.jsonl> --model <name>` routes
predictions through `tools/answer_formatter.py` (per-family extractors —
\boxed{}, MCQ parse, preamble strip, code-fence unwrap, entity+URL extract)
then through the family judge, then writes a markdown delta-vs-baseline
report.

### 4.2 Baselines

- Typhoon-2-8B-Instruct (arxiv 2412.13702v2): ThaiExam 0.512, M3Exam 0.475,
  IFEval-TH 0.726, MT-Bench-TH 5.74, MATH 0.490, GSM8K 0.81
- Typhoon-2-70B-Instruct: ThaiExam 0.634, M3Exam 0.623, BFCL-TH 0.709
- OpenThaiGPT-1.5 7B: ThaiExam 0.520, M3Exam 0.540, OpenThaiEval 0.658
- OpenThaiGPT-1.5 72B: ThaiExam 0.641

For benchmarks where neither paper publishes scores (AIME-TH, LiveCodeBench-TH,
HotpotQA), we will run the Typhoon-2-8B and OpenThaiGPT-1.5 7B checkpoints
ourselves through `tools/run_inference.py` to establish baselines.

### 4.3 LLM-judge methodology

For benchmarks requiring LLM judgment (MT-Bench rubric scoring, OpenThaiEval
analytic answers, HotpotQA answer equivalence), we use gpt-5.5-xhigh as the
judge with:

- Temperature 0.0
- Position-swap for pairwise (run A-vs-B and B-vs-A; tie if disagree)
- Rubric forbids reference to model identity (mitigates self-preference)
- On-disk cache keyed by sha256(prompt + response + rubric)
- Cohen's κ ≥ 0.85 target on a 100-judgment human-spot-check subset

### 4.4 Determinism path

For benchmarks with deterministic ground truth (AIME, MATH500, OpenThaiEval
MCQ, IFEval verifiable constraints, LiveCodeBench unit tests), sympy /
unit-test / regex paths run BEFORE any LLM call — the LLM judge is fallback
only when deterministic methods can't decide.

## 5. Data quality measurements

`tools/dataset_quality_analysis.py` produces:

- Per-family record counts ours vs public
- Prompt-length distribution per family (p10 / median / p90)
- Type-token ratio (TTR) for user prompts and assistant responses
  (>0.05 threshold to detect templated repetition)
- Difficulty + language balance
- Estimated training-token budget (current: 911k tokens — appropriate for
  SFT, in Typhoon-2-8B SFT range)
- Side-by-side prompt samples

## 6. Reproducibility

- Schema validates every record (`jsonschema.Draft202012Validator`).
- Audit is re-runnable: `python tools/audit_run.py --no-prosody` →
  4,277/4,288 pass (99.74%).
- Pipeline is deterministic: `python tools/repair_pipeline.py` reproduces
  the full corpus state from sources.
- Snapshot at `tools/_locked_snapshot/` is the canonical artifact for the
  paper's data appendix.

## 7. Threats to validity

- **Sync interference**: external OneDrive sync silently reverted dataset
  files three times during construction. Mitigated by `tools/lock_state.py`
  setting files read-only. Reproduction requires either disabling sync or
  re-running `lock_state.py`.
- **Source language**: most hand-authored records' supporting facts are
  verifiable on English Wikipedia; Thai Wikipedia coverage is patchier.
  9 of 30 initial hotpotqa URLs were broken because the Thai-language wiki
  article didn't exist at the slug we guessed — switched to English.
- **gpt-5.5-xhigh self-preference**: the judge is the same model family as
  the generator. Mitigated by temperature 0.0 + rubric instructions + the
  per-family deterministic verifier running first.
- **644k–911k tokens** is small for a 35B model. We are betting that
  *quality* + *teacher-loop self-correction signal* matter more than scale,
  which the ablation will test.
