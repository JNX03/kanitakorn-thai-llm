# Quality Upgrade Audit

Upgrade version: `quality_gate_2026_05_08_v1`

Bulk generation is paused. The corpus may not resume toward 20k until a new 100-item mixed audit batch passes the stricter gate.

## Applied Changes

- Difficulty labels are recalibrated at packaging time; simple drills are no longer reported as hard.
- Quality scores are capped and annotated with reviewer notes instead of preserving inflated self-scores.
- One-source HotpotQA-agentic items are quarantined from accepted shards.
- Embedding similarity is explicitly marked `not_run` with null max values.
- Rejection quota policy is recorded for all future accepted shards.

## Current Accepted Difficulty Counts

| Family | Easy | Medium | Hard | Olympiad | Adversarial |
|---|---:|---:|---:|---:|---:|
| aime_th | 0 | 273 | 837 | 186 | 0 |
| hotpotqa_agentic | 0 | 0 | 31 | 0 | 0 |
| ifeval_ifbench | 0 | 36 | 439 | 0 | 221 |
| livecodebench_th | 27 | 428 | 241 | 0 | 0 |
| math500_th | 127 | 344 | 226 | 0 | 0 |
| mt_bench | 0 | 10 | 86 | 0 | 0 |
| openthaieval | 27 | 425 | 244 | 0 | 0 |
| teacher_loop_th | 0 | 50 | 0 | 0 | 0 |

Accepted items with reviewer notes: 4258 / 4258

## Future 100-Item Mixed Audit Gate

- Mixed audit accepted items: 1270 / 100.
- Mixed audit passed: True.
- AIME-TH: at least 70% of new accepted items must need three or more reasoning steps.
- MATH500-TH: include Level 4-5 style items; easy items are allowed only for coverage.
- LiveCodeBench-TH: require harder original tasks with public, hidden, and randomized tests where possible.
- IFEval/IFBench: reduce repeated line/keyword templates and add Thai-specific nested constraints.
- HotpotQA-agentic: require two or more sources and two or more supporting facts.
- OpenThaiEval: require plausible distractors and subtle inference for MC items.
- Rejection/quarantine log must include at least one rejected/quarantined candidate for every accepted production shard.
