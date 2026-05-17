# Quality Gate Manual Audit Check

Date: 2026-05-08

The quality upgrade pass is complete. The upgraded 100-item mixed audit gate has passed with 315 accepted mixed-audit items under the stricter standards, and production generation has resumed under those rules.

## Executable Checks

- Full packaging and verification command passed: `python tools\package_and_verify.py`.
- Checkpoint refreshed with: `python tools\write_checkpoint.py`.
- Targeted metadata scan covered all emitted train and validation JSONL rows.

## Targeted Scan Results

| Check | Result |
|---|---:|
| Accepted rows scanned | 3253 |
| Metadata errors found | 0 |
| Rejected/quarantined rows | 186 |
| Reject decisions | 93 |
| Quarantine decisions | 93 |
| HotpotQA-agentic rows checked | 16 |
| IFEval/IFBench non-deterministic verifier rows | 0 |
| AIME quality-gate train rows checked | 39 |
| AIME quality-gate multi-step estimate | 35 |
| AIME quality-gate multi-step share | 0.897 |

## Confirmed Upgrade Properties

- `embedding_similarity_status` is `not_run` and `embedding_similarity_max` is `null` for accepted rows.
- Quality scores are capped below inflated perfect self-scores and include `calibration_version`, `difficulty_calibration`, and `reviewer_notes`.
- One-source HotpotQA-agentic rows are excluded from accepted shards by the packaging quality gate.
- Accepted HotpotQA-agentic rows include at least two sources and at least two required supporting facts.
- IFEval/IFBench rows use deterministic `regex`, `json_schema`, or `exact_match` verifiers.
- The rejection quota policy is recorded, and the current audit log contains both rejects and quarantines.

## Current Limitation

Neural embedding similarity remains intentionally marked as not run. A production release should run embedding similarity against the protected benchmark registry before large-scale training.
