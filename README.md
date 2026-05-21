# Kanitakorn — Thai LLM Campaign (≤14B)

Active 7-day campaign to fine-tune a Thai LLM that beats every existing Thai LLM at ≤14B params on 12 benchmarks.

## Targets

| Benchmark | Target | Current 8B SOTA |
|-----------|-------:|----------------:|
| AIME24-TH | >15 | OpenThaiGPT-1.6-72B: 6.67 |
| AIME24 | >25 | OpenThaiGPT-1.6-72B: 23.33 |
| MATH500-TH | >56 | OpenThaiGPT-1.6-72B: 43.2 |
| MATH500 | >82 | OpenThaiGPT-1.6-72B: 82.0 |
| LiveCodeBench-TH | >35 | OpenThaiGPT-1.6-72B: 32.43 |
| LiveCodeBench | >60 | OpenThaiGPT-1.6-72B: 54.21 |
| OpenThaiEval | >80 | OpenThaiGPT-1.6-72B: 78.7 |
| HotpotQA | >46 | Typhoon-S TH: 37 |
| IFEval (en) | >57 | n/a |
| MT-BENCH | >85 | Typhoon-S TH: 78.9 |
| THAIEXAM | >75 | OpenThaiGPT-1.5: 0.520, Pathumma: 0.513 |
| IFEval-TH | >82 | Typhoon-S: 76.45 |

## Approach

Strategic insight from research (Phase 1):
- **DeepSeek-R1-Distill-Qwen-14B** already scores MATH500=93.9, AIME24=69.7 out of the box (MIT license).
- The hard targets are **Thai language ability**, not math/code.
- Recipe: keep math reasoning intact while adding Thai → minimal-disruption Thai SFT.

## Architecture

- **Base**: `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B` (15B params, MIT)
- **Method**: QLoRA (4-bit nf4) + LoRA r=16 on attention+MLP layers
- **Data**: ~4528 Thai SFT (kanitakorn-th-sft-v4-distill, includes R1-distilled traces) + 6000 EN mix (MetaMathQA + OpenCoder + OpenHermes)
- **Compute**: 2× A100 40GB vast.ai (interruptible, ~$0.43/hr)

## Files

| File | Purpose |
|------|---------|
| `train_2xa100_interruptible.py` | DDP-aware SFT with QLoRA + auto-resume |
| `boot_and_train.sh` | Idempotent pipeline (deps → download → rebuild → eval → train → push) |
| `eval_self_consistency.py` | Batched eval with cross-record batching |
| `eval_vllm.py` | vLLM-based fast eval (~5-10x speedup) |
| `eval_deepconf.py` | Confidence-weighted Maj@64 (+20pp on AIME24) |
| `merge_sweep.py` | Task-arithmetic LoRA merge sweep for Stage C |
| `tools/hf_auto_publish.py` | Auto-version + push adapter to HF with model card |
| `tools/distill_from_teacher.py` | Generate R1 CoT traces via OpenRouter |
| `tools/rebuild_for_base.py` | Re-render SFT records with target base's chat template |
| `CAMPAIGN_7DAY.md` | Full campaign plan + reality check |
| `RESUME_PROTOCOL.md` | Context-survival doc for picking up where we left off |
| `VERSION_LOG.md` | Per-version: base, dataset, hyper, eval scores |

## HF artifacts

- Dataset: [Jnx03/kanitakorn-th-sft-v4-distill](https://huggingface.co/datasets/Jnx03/kanitakorn-th-sft-v4-distill) (4267 SFT + 291 R1-distilled traces)
- Model: pushed via `tools/hf_auto_publish.py` to `Jnx03/kanitakorn-r1d-qwen14b-*` series

## License

MIT (inherited from DeepSeek-R1-Distill-Qwen-14B). Code in this repo: MIT.

## References

- DeepSeek-R1 paper: [arXiv:2501.12948](https://arxiv.org/html/2501.12948v1)
- Typhoon-2 paper (Thai LLM SOTA): [arXiv:2412.13702](https://arxiv.org/abs/2412.13702)
- DeepConf (confidence-weighted Maj@N): see `eval_deepconf.py` references
- SimpleTIR (Python tool-use for math): [arXiv:2509.02479](https://arxiv.org/pdf/2509.02479)
