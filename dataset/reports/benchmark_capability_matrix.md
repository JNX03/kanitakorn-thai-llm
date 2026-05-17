# Benchmark Capability Matrix

This is a verified partial run, not the requested 20,000 item full corpus.

| Family | Source findings | Target capabilities | Dataset response |
|---|---|---|---|
| AIME24/25-TH | math-ai AIME24/AIME25 expose 30 test rows each with problem/answer or solution fields and Apache-2.0 metadata. | Olympiad-style integer answer extraction, CRT, counting, probability. | Original Thai competition items only; AIME prompts/answers are blacklist-only. |
| MATH500-TH | OpenAI simple-evals documents MATH-500 as the newer IID MATH subset for newer model evals; MATH source is MIT. | Algebra, geometry, probability with concise solutions. | Original Thai MATH-like items with symbolic checks. |
| LiveCodeBench-TH | LiveCodeBench covers code generation, self-repair, execution, and output prediction; Thai HF set exposes question fields and large private tests. | Thai problem comprehension, debugging, edge-case testing, execution tracing. | Original tasks; no contest platform statements copied. |
| OpenThaiEval | iapp/openthaieval reports 1,232 questions across 17 exam types, with MC/explanation metadata. | Thai reading, NLI, professional QA. | Self-contained or source-backed Thai exam-style items. |
| MT-Bench Thai/English | MT-Bench is multi-turn LLM-as-judge evaluation; Thai set has 91 two-turn rows. | Multi-turn consistency, revision, natural Thai style. | Original two-turn dialogues with judge rubrics. |
| IFEval/IFBench | IFEval uses verifiable instructions; IFBench stresses unseen constraint generalization. | Deterministic format, keyword, paragraph, JSON, and punctuation constraints. | Original Thai/English verifiable tasks with checkers. |
| HotpotQA-style | HotpotQA requires multi-document reasoning and sentence-level supporting facts. | Evidence linking, source discipline, ambiguity control. | Source-grounded Thai/bilingual multi-hop questions with supporting facts. |

Primary sources: Hugging Face dataset cards for Thai MT-Bench, IFEval-TH, AIME24/25, LiveCodeBench-TH, OpenThaiEval; arXiv 2306.05685, 2311.07911, 1809.09600, 2403.07974; Google Gemma 4 blog; Qwen HF model card.
