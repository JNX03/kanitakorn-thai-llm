# Setup Report

This run produced a verified partial corpus rather than padding toward the requested 20,000-item target. The full target remains 20,000 train / 2,000 validation with the requested mixture, but this shard is a pilot-quality audited subset.

## 1. Benchmark Capability Matrix

See `benchmark_capability_matrix.md`. The working capability targets are Thai olympiad math, Thai MATH-style breadth, Thai coding generation/debugging/execution, OpenThaiEval-style exam/NLI/professional QA, bilingual MT-Bench dialogue quality, deterministic IF constraints, and source-grounded multi-hop QA.

## 2. Contamination-Risk Plan

Official benchmark prompts, answer keys, public examples, and private tests are evaluation-only. The protected registry uses normalized exact matching, Thai/English whitespace normalization, character 5-gram overlap, SimHash, numeric-structure review, and problem-statement review. Full production should add semantic embedding search over a complete offline registry including full LiveCodeBench private payloads.

## 3. Dataset Mixture and Target Size

Full target remains:
- 3,600 AIME-TH train / 360 validation
- 3,000 MATH500-TH train / 300 validation
- 3,000 LiveCodeBench-TH train / 300 validation
- 3,000 OpenThaiEval train / 300 validation
- 2,400 MT-Bench train / 240 validation
- 3,000 IFEval/IFBench train / 300 validation
- 2,000 HotpotQA-agentic train / 200 validation

Completed after continuation shard 001: 35 train / 14 validation, balanced at 7 accepted items per family. The continuation added 21 train and 7 validation items without replacing shard 000.

## 4. Verifier Plan

Math uses symbolic/arithmetic assertions and independent solution outlines. Coding uses public tests plus randomized brute force where applicable. Instruction following uses JSON/schema or regex-like deterministic checks. OpenThaiEval uses exact-match or retrieval evidence. MT-Bench uses bilingual rubric checks. HotpotQA-agentic uses source-backed required supporting facts.

## 5. Manual Generation Protocol

Items were authored deliberately one by one. Code was used only for verification, contamination checks, schema validation, sharding, and reporting. Low-confidence factual or low-originality drafts were rejected or quarantined.

## 6. Subagent-Use Protocol

One Codex subagent was used only for risk review. No subagent-authored item was accepted directly. The main agent authored, reviewed, verified, and accepted every included item.

## 7. One-By-One Review Checklist

Each accepted item passed originality, benchmark-leakage, Thai naturalness, correctness, concise reasoning, verifier validity, metadata completeness, difficulty label, benchmark-family label, and transferable-skill checks.

## 8. Initial Source List

Primary references inspected include Hugging Face cards for Thai MT-Bench, IFEval-TH, AIME24/25, LiveCodeBench-TH, OpenThaiEval, Qwen/Qwen3.6-35B-A3B, Google Gemma 4 pages, arXiv 2306.05685, 2311.07911, 1809.09600, 2403.07974, OpenAI simple-evals, UNESCO, Britannica, and the Bank of Thailand.

## 9. Planned Shard Structure

The requested shard structure was created under `dataset/train`, `dataset/validation`, `dataset/reports`, and `dataset/verifiers`.
