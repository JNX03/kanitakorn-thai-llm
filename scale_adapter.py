"""Scale a LoRA adapter's weights by alpha at the safetensors level.

This effectively applies the adapter at fractional strength without needing
PEFT's set_adapter_scale at inference time. Useful when an adapter trained
at high LR damaged the base — we can scale down to recover.

Math: trained adapter = ΔW. Loading scaled at α gives ΔW' = α·ΔW.
At inference, model output = base + ΔW' = base + α·ΔW.
α=1.0 → original, α=0.5 → half-strength, α=0.0 → just base.

Usage:
    python3 scale_adapter.py \\
        --in-dir runs/stage1_v2/final \\
        --out-dir runs/stage1_v2_scaled_0.5 \\
        --alpha 0.5
"""
import argparse, shutil, json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--alpha", type=float, required=True, help="0.0 to 1.0+ (1.0 = no change)")
    args = ap.parse_args()

    src = Path(args.in_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Scale adapter_model.safetensors
    print(f"[scale] loading {src}/adapter_model.safetensors")
    sd = load_file(src / "adapter_model.safetensors")
    scaled = {}
    for k, v in sd.items():
        if "lora_A" in k or "lora_B" in k:
            # Scale only the trainable LoRA matrices.
            # Note: ΔW = B @ A * scale, so scaling either A or B suffices.
            # Convention: scale lora_B by alpha.
            if "lora_B" in k:
                scaled[k] = (v.float() * args.alpha).to(v.dtype)
            else:
                scaled[k] = v.clone()
        else:
            scaled[k] = v.clone()
    save_file(scaled, out / "adapter_model.safetensors")
    print(f"[scale] saved {out}/adapter_model.safetensors (alpha={args.alpha})")

    # Copy config + tokenizer
    for fname in ["adapter_config.json", "tokenizer.json", "tokenizer_config.json",
                  "chat_template.jinja", "training_args.bin", "README.md", "special_tokens_map.json"]:
        sp = src / fname
        if sp.exists():
            shutil.copy(sp, out / fname)
    print(f"[done] scaled adapter at {out}")

if __name__ == "__main__":
    main()
