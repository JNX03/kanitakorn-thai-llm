"""Qwen3.6-35B-A3B SFT training script for the A100.

Usage on the A100 after `bash bootstrap_a100.sh`:

    # Main dataset (default)
    python tools/train_sft.py \\
        --base Qwen/Qwen3.6-35B-A3B-Instruct \\
        --manifest dataset/sft_ready/manifest.json \\
        --output runs/qwen3.6-35b-a3b-th-sft-v1

    # Backup variant (if main run underperforms)
    python tools/train_sft.py \\
        --base Qwen/Qwen3.6-35B-A3B-Instruct \\
        --manifest dataset/sft_ready_variants/tutor/manifest.json \\
        --output runs/qwen3.6-35b-a3b-th-tutor

    # Smaller model fallback (Gemma 4 or 7B base, for fast iteration on T4)
    python tools/train_sft.py \\
        --base Qwen/Qwen2.5-7B-Instruct \\
        --manifest dataset/sft_ready/manifest.json \\
        --output runs/qwen2.5-7b-th-sft \\
        --max-steps 500

Uses PEFT (LoRA) by default — fits a 35B model in 80GB VRAM. Override with
--full-finetune for full SFT (needs FSDP / multi-GPU).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, help="HuggingFace base model id (e.g. Qwen/Qwen3.6-35B-A3B-Instruct)")
    parser.add_argument("--manifest", required=True, help="Path to sft_ready manifest.json (defines weights + files)")
    parser.add_argument("--output", required=True, help="Output directory for checkpoints")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--per-device-batch", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=-1, help="cap total steps (for quick smoke runs)")
    parser.add_argument("--full-finetune", action="store_true", help="disable LoRA — needs multi-GPU FSDP")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    # Heavy imports deferred so --help is fast.
    try:
        import torch  # type: ignore
        from datasets import Dataset, concatenate_datasets, load_dataset  # type: ignore
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
        from trl import SFTConfig, SFTTrainer  # type: ignore
        if not args.full_finetune:
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training  # type: ignore
    except ImportError as e:
        print(f"missing dependency: {e}")
        print("run bash bootstrap_a100.sh first to install requirements")
        return 2

    print(f"CUDA: {torch.cuda.is_available()}; device count: {torch.cuda.device_count()}")
    if not torch.cuda.is_available():
        print("WARN: no CUDA detected — training on CPU will be glacial. Continue anyway? (Ctrl-C to abort, 5s)")
        import time
        time.sleep(5)

    print(f"\nLoading manifest from {args.manifest}")
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    weights = manifest.get("weights", {})
    print(f"families: {list(weights.keys())}; weights: {weights}")

    # Load each family's SFT file as a HF Dataset, then weighted-sample.
    root = Path(args.manifest).resolve().parent.parent
    family_datasets = {}
    for fam, paths in manifest["sft_files"].items():
        if paths.get("train"):
            full_path = root / paths["train"]
            if full_path.exists():
                ds = load_dataset("json", data_files=str(full_path), split="train")
                family_datasets[fam] = ds
                print(f"  {fam:<22} {len(ds):>6} train records")

    # Weighted sampling: replicate each family by ceil(weight * total_target_size / family_size)
    target_size = sum(len(ds) for ds in family_datasets.values()) * 2  # 2x for upweighted families
    weighted_parts = []
    for fam, ds in family_datasets.items():
        w = weights.get(fam, 1.0 / len(family_datasets))
        n_repeat = max(1, int(round(w * target_size / max(len(ds), 1))))
        weighted_parts.append(ds.shuffle(seed=args.seed).select(range(min(len(ds), n_repeat * len(ds)))))
    full_ds = concatenate_datasets(weighted_parts).shuffle(seed=args.seed)
    print(f"\nTotal training records after weighted sampling: {len(full_ds)}")

    print(f"\nLoading tokenizer + model: {args.base}")
    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model_kwargs = {
        "torch_dtype": torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        "device_map": "auto",
        "trust_remote_code": True,
    }
    if not args.full_finetune:
        # 4-bit quantization for LoRA → 35B fits in 80GB easily.
        try:
            from transformers import BitsAndBytesConfig  # type: ignore
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        except ImportError:
            print("WARN: bitsandbytes not available — falling back to bf16 LoRA (more VRAM)")

    model = AutoModelForCausalLM.from_pretrained(args.base, **model_kwargs)

    if not args.full_finetune:
        model = prepare_model_for_kbit_training(model)
        lora_cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()

    # Each record carries `text` (Qwen chat template); use SFTTrainer's text-only flow.
    print(f"\nTraining: epochs={args.epochs}, lr={args.lr}, per_device_batch={args.per_device_batch}, grad_accum={args.grad_accum}")
    sft_cfg = SFTConfig(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        max_seq_length=args.max_seq_len,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        report_to="none",
        warmup_ratio=0.03,
        weight_decay=0.01,
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tok,
        train_dataset=full_ds,
        args=sft_cfg,
    )
    trainer.train()
    trainer.save_model(f"{args.output}/final")
    tok.save_pretrained(f"{args.output}/final")
    print(f"\nDone. Saved final checkpoint to {args.output}/final")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
