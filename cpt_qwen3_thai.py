"""Continued Pretraining (CPT) of Qwen3-14B on Thai corpus.

Autoregressive LM loss on Thai text. Uses QLoRA r=16 for memory efficiency.
Streams Fineweb2-Thai (or fallback Thai dataset) — no local download needed.

Recipe:
- Base: Qwen3-14B
- Adapter: QLoRA 4-bit + LoRA r=16 (target_modules: all linear)
- LR: 1e-5 (CPT — slightly higher than SFT)
- Context: 2048 tokens
- Batch: 2 × grad_accum 4 = effective 8
- Save every 50 steps for preemption safety

Goal: deeply integrate Thai language knowledge into Qwen3-14B so subsequent
SFT can leverage stronger Thai foundation, NOT damage base.

Launch:
    CUDA_VISIBLE_DEVICES=0 python3 cpt_qwen3_thai.py
"""
import os, json
from pathlib import Path

import torch
from datasets import load_dataset, IterableDataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer

BASE = os.environ.get("CPT_BASE", "Qwen/Qwen3-14B")
OUT = Path(os.environ.get("CPT_OUT", "/root/kanitakorn/runs/cpt_qwen3_thai"))
MAX_STEPS = int(os.environ.get("CPT_STEPS", "1500"))
LR = float(os.environ.get("CPT_LR", "1e-5"))
LORA_R = int(os.environ.get("CPT_LORA_R", "16"))
MAX_LEN = int(os.environ.get("CPT_MAX_LEN", "2048"))

print(f"[cpt] base={BASE} out={OUT} steps={MAX_STEPS} lr={LR} r={LORA_R} max_len={MAX_LEN}")

OUT.mkdir(parents=True, exist_ok=True)

# ---- Tokenizer ----
print("[tok] loading...")
tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
if tok.pad_token is None: tok.pad_token = tok.eos_token

# ---- Streaming Thai corpus ----
# Fineweb2-Thai is the cleanest Thai web corpus (used by Typhoon 2, ThaiLLM)
print("[data] streaming Thai corpus...")
sources_to_try = [
    ("HuggingFaceFW/fineweb-2", "tha_Thai_removed", "train"),  # primary
    ("HuggingFaceFW/fineweb-2", "tha_Thai", "train"),  # fallback
    ("uonlp/CulturaX", "th", "train"),  # backup
]
ds = None
for repo, config, split in sources_to_try:
    try:
        print(f"[data]   trying {repo}/{config}/{split}...")
        ds = load_dataset(repo, config, split=split, streaming=True,
                          trust_remote_code=True)
        # Sanity check — pull first record
        first = next(iter(ds))
        text_field = "text" if "text" in first else "content"
        if text_field in first and any('฀' <= c <= '๿' for c in first[text_field][:1000]):
            print(f"[data]   ✓ {repo}/{config} works, text field: {text_field}")
            break
        else:
            ds = None
    except Exception as e:
        print(f"[data]   skip {repo}/{config}: {str(e)[:120]}")
        ds = None

if ds is None:
    raise SystemExit("Could not load any Thai corpus")

# Convert to format SFTTrainer expects (with 'text' field)
def text_iter():
    n = 0
    for r in ds:
        text = r.get("text") or r.get("content", "")
        if not text or len(text) < 200: continue
        # Filter: must contain Thai
        if not any('฀' <= c <= '๿' for c in text[:500]): continue
        yield {"text": text[:8000]}  # cap length
        n += 1
        if n >= MAX_STEPS * 16: break  # enough for max_steps

stream_ds = IterableDataset.from_generator(text_iter)

# ---- Model ----
print("[model] loading 4-bit...")
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.bfloat16,
                         bnb_4bit_use_double_quant=True)
model = AutoModelForCausalLM.from_pretrained(
    BASE, quantization_config=bnb, device_map="auto",
    trust_remote_code=True, attn_implementation="sdpa",
)
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
model.config.use_cache = False

lora = LoraConfig(
    r=LORA_R, lora_alpha=LORA_R*2, lora_dropout=0.05,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    bias="none", task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora)
if not hasattr(model.base_model.model, "warnings_issued"):
    model.base_model.model.warnings_issued = {}
model.print_trainable_parameters()

# ---- Find last checkpoint for resume ----
def find_last_ckpt(d):
    if not d.exists(): return None
    cs = sorted(d.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
    return str(cs[-1]) if cs else None

resume = find_last_ckpt(OUT)
print(f"[ckpt] resume={resume}")

cfg = SFTConfig(
    output_dir=str(OUT),
    max_steps=MAX_STEPS,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=LR,
    warmup_ratio=0.02,
    lr_scheduler_type="cosine",
    logging_steps=20,
    save_steps=int(os.environ.get("CPT_SAVE_STEPS", "100")),
    save_total_limit=3,
    bf16=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    optim="paged_adamw_8bit",
    max_length=MAX_LEN,
    packing=False,  # streaming — packing complicates things
    dataset_text_field="text",
    report_to="none",
    seed=2026,
    dataloader_num_workers=0,  # streaming dataset
    max_grad_norm=1.0,
)

trainer = SFTTrainer(model=model, args=cfg, train_dataset=stream_ds,
                      processing_class=tok)
print("[cpt] starting...")
trainer.train(resume_from_checkpoint=resume)
trainer.save_model(str(OUT / "final"))
tok.save_pretrained(str(OUT / "final"))
print("[cpt] DONE")
