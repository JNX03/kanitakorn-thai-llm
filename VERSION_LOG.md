# Kanitakorn Version Log

One row per published adapter. Sorted oldest-first.

Campaign targets recap: THAIEXAM>75, MATH500>82, AIME24>25 (top-3); plus 9 others (see CAMPAIGN_7DAY.md).

| Date | Repo | Stage | Base | LR | R | Epochs | THAIEXAM | MATH500 | AIME24 | Notes |
|------|------|-------|------|---:|--:|-------:|---------:|--------:|-------:|-------|
| 2026-05-21 | [kanitakorn-th-sft-v4-distill](https://huggingface.co/datasets/Jnx03/kanitakorn-th-sft-v4-distill) | dataset | mixed | — | — | — | — | — | — | 4267 SFT + 291 R1-distilled traces (math/AIME/OpenThaiEval in DeepSeek format) |
| 2026-05-21 | kanitakorn-r1d14b-stage1 (training NOW) | stage1 | DeepSeek-R1-Distill-Qwen-14B | 1e-5 | 16 | 2 | TBD | TBD | TBD | QLoRA nf4, 15060 records (9060 TH+6000 EN), single GPU after ECC error on GPU 1. ~3766 steps. |
