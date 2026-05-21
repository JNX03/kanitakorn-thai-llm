"""Unsloth fast-path SFT — 2-5x throughput vs standard PEFT on QLoRA.

Same env-vars as train_2xa100_interruptible.py. Use this when speed matters
more than DDP (unsloth is single-GPU but >2x faster, so for our 14B+QLoRA
case on 2x40GB we should benchmark vs DDP to pick).

Launch:
    SFT_BASE=deepseek-ai/DeepSeek-R1-Distill-Qwen-14B \\
    SFT_OUT=/root/kanitakorn/runs/unsloth_v1 \\
    python3 train_unsloth_fast.py

Note: unsloth tends to require its own torch+transformers versions. If
installation fails, fall back to train_2xa100_interruptible.py.
"""
import os, json, random
from pathlib import Path

import torch
from datasets import Dataset, concatenate_datasets, load_dataset

# Unsloth import — fail-fast if not installed
try:
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template
except ImportError:
    raise SystemExit("Unsloth not installed. `pip install 'unsloth[cu124]'` first.")

from trl import SFTConfig, SFTTrainer
from transformers import AutoTokenizer

PROJECT = Path(os.environ.get("SFT_PROJECT", "/root/kanitakorn"))
MANIFEST = Path(os.environ.get("SFT_MANIFEST", PROJECT / "dataset/sft_ready/manifest.json"))
BASE = os.environ.get("SFT_BASE", "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B")
OUT = Path(os.environ.get("SFT_OUT", PROJECT / "runs/unsloth_v1"))
EPOCHS = float(os.environ.get("SFT_EPOCHS", "3"))
LORA_R = int(os.environ.get("SFT_LORA_R", "16"))
SEED = int(os.environ.get("SFT_SEED", "2026"))
MAX_SEQ = int(os.environ.get("SFT_MAX_SEQ", "2048"))

print(f"[unsloth] base={BASE} out={OUT} epochs={EPOCHS} r={LORA_R} max_seq={MAX_SEQ}")

# Load model with unsloth's optimized kernels + 4-bit
model, tok = FastLanguageModel.from_pretrained(
    model_name=BASE,
    max_seq_length=MAX_SEQ,
    dtype=None,           # auto
    load_in_4bit=True,
)

# Add LoRA via unsloth
model = FastLanguageModel.get_peft_model(
    model, r=LORA_R, target_modules=[
        "q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"
    ],
    lora_alpha=LORA_R*2, lora_dropout=0.05, bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=SEED, max_seq_length=MAX_SEQ,
)

# Load data
manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
weights = manifest["weights"]
files = manifest["sft_files"]
DATA_ROOT = MANIFEST.parent.parent
recs = []
rng = random.Random(SEED)
for fam, paths in files.items():
    fp = DATA_ROOT / paths["train"]
    fam_recs = []
    for line in fp.open(encoding="utf-8"):
        if not line.strip(): continue
        r = json.loads(line)
        if "text" in r and len(r["text"]) < 8000:
            fam_recs.append({"text": r["text"]})
    # Weighted sample
    w = weights[fam]
    target = max(int(len(fam_recs) * w * 3 / max(weights.values())), len(fam_recs))
    if target > len(fam_recs):
        idx = [rng.randint(0, len(fam_recs)-1) for _ in range(target)]
    else:
        idx = rng.sample(range(len(fam_recs)), target)
    recs.extend([fam_recs[i] for i in idx])
    print(f"  {fam}: {len(fam_recs)} loaded → {target} sampled (w={w:.3f})")
random.Random(SEED).shuffle(recs)
print(f"[data] total train={len(recs)}")
ds = Dataset.from_list(recs)

OUT.mkdir(parents=True, exist_ok=True)
cfg = SFTConfig(
    output_dir=str(OUT),
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=1e-5,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    logging_steps=20,
    save_steps=200,
    save_total_limit=3,
    bf16=True,
    optim="paged_adamw_8bit",
    max_length=MAX_SEQ,
    packing=False,
    dataset_text_field="text",
    report_to="none",
    seed=SEED,
    dataloader_num_workers=2,
)

def find_last_ckpt(d):
    if not d.exists(): return None
    cs = sorted(d.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
    return str(cs[-1]) if cs else None

resume = find_last_ckpt(OUT)
print(f"[ckpt] resume={resume}")

trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds, processing_class=tok)
print("[train] starting")
trainer.train(resume_from_checkpoint=resume)
trainer.save_model(str(OUT / "final"))
tok.save_pretrained(str(OUT / "final"))
print("[done]")
