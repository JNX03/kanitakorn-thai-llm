# Thai Linguistic QA Report

Checks performed:
- Thai prompts were reviewed for native order, register consistency, and non-literal English translation.
- Formal exam items use concise school-evaluation Thai.
- Coding items use common Thai competitive-programming wording while preserving technical terms such as prefix sum, hashmap, endpoint, and API where natural.
- MT-Bench-style items use practical Thai rather than generic assistant prose.
- IFEval items intentionally use rigid punctuation or format only when required by the verifier.

Findings:
- No accepted Thai item contains unresolved register mismatch.
- Code-switching appears only in bilingual business/coding contexts.
- Buddhist Era dates and Thai numerals are now included in the upgraded IFEval/IFBench audit items with deterministic checks.
