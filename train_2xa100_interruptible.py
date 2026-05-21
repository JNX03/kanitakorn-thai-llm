"""Multi-GPU DDP + interruptible-safe SFT for 7-day campaign on 2x A100 40GB.

Loads from dataset/sft_ready/manifest.json with weighted sampling per family.
Saves checkpoints every 200 steps for fast recovery on vast.ai preemption.
Resumes automatically from the latest checkpoint in OUT/.

Launch (2x A100):
    torchrun --nproc_per_node=2 train_2xa100_interruptible.py

Launch (1 GPU fallback):
    python3 train_2xa100_interruptible.py

Env vars:
    SFT_BASE  = base model (default Qwen/Qwen3-8B-Base)
    SFT_OUT   = output dir   (default /workspace/kanitakorn/runs/campaign_v1)
    SFT_MANIFEST = manifest path (default dataset/sft_ready/manifest.json)
    SFT_EPOCHS = total epochs (default 3)
    SFT_LORA_R = lora rank (default 32 — larger for more capacity)
    SFT_MIX_EN = 1 to mix in English+math+code via streaming HF datasets (default 1)
"""
import os, json, random, math, glob, sys
from pathlib import Path

import torch
from datasets import Dataset, concatenate_datasets, load_dataset, IterableDataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer

# ---- Config ---------------------------------------------------------------
PROJECT = Path(os.environ.get("SFT_PROJECT", "/workspace/kanitakorn"))
MANIFEST = Path(os.environ.get("SFT_MANIFEST", PROJECT / "dataset/sft_ready/manifest.json"))
BASE = os.environ.get("SFT_BASE", "Qwen/Qwen3-8B-Base")
OUT = Path(os.environ.get("SFT_OUT", PROJECT / "runs/campaign_v1"))
EPOCHS = float(os.environ.get("SFT_EPOCHS", "3"))
LORA_R = int(os.environ.get("SFT_LORA_R", "32"))
MIX_EN = os.environ.get("SFT_MIX_EN", "1") == "1"
SEED = int(os.environ.get("SFT_SEED", "2026"))
# QLoRA: auto-enable for bases that won't fit in bf16 on 40GB GPU
QLORA_ENV = os.environ.get("SFT_QLORA", "auto").lower()
def _auto_qlora(base_name: str) -> bool:
    s = base_name.lower()
    # 14B+ bases need 4-bit on 2x A100 40GB
    return any(k in s for k in ("14b", "14B", "32b", "32B", "70b", "70B"))
USE_QLORA = (QLORA_ENV == "1") or (QLORA_ENV == "auto" and _auto_qlora(BASE))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))
RANK = int(os.environ.get("RANK", "0"))

is_main = RANK == 0
def log(msg):
    if is_main: print(msg, flush=True)

log(f"[cfg] base={BASE} out={OUT} epochs={EPOCHS} lora_r={LORA_R} mix_en={MIX_EN} qlora={USE_QLORA}")
log(f"[cfg] world_size={WORLD_SIZE} local_rank={LOCAL_RANK} rank={RANK}")

# ---- Tokenizer (used for English data formatting) -------------------------
log("[tok] loading tokenizer + chat template")
tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
if tok.pad_token is None: tok.pad_token = tok.eos_token
if not getattr(tok, "chat_template", None):
    qwen_tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", trust_remote_code=True)
    tok.chat_template = qwen_tok.chat_template
    log("[tok] borrowed Qwen3-8B chat template")

# ---- Load manifest + Thai SFT-ready data ----------------------------------
log("[data] loading manifest")
manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
weights = manifest["weights"]
files = manifest["sft_files"]
# manifest paths are relative to dataset/ (manifest.json's parent's parent)
DATA_ROOT = MANIFEST.parent.parent

def load_family(fam: str) -> Dataset:
    fp = DATA_ROOT / files[fam]["train"]
    recs = []
    for line in fp.open(encoding="utf-8"):
        if not line.strip(): continue
        r = json.loads(line)
        text = r.get("text")
        if text and len(text) < 8000:  # safety guard
            recs.append({"text": text})
    return Dataset.from_list(recs)

family_datasets = {}
total_thai = 0
for fam in files:
    d = load_family(fam)
    family_datasets[fam] = d
    total_thai += len(d)
    log(f"[data]   {fam}: {len(d)} (weight={weights[fam]:.3f})")

# Weighted sampling — upsample/downsample each family to match weight target.
TARGET_THAI = sum(len(d) for d in family_datasets.values()) * int(EPOCHS)
sampled = []
rng = random.Random(SEED)
for fam, d in family_datasets.items():
    w = weights[fam]
    target_count = int(TARGET_THAI * w)
    if len(d) == 0: continue
    if target_count <= len(d):
        idx = rng.sample(range(len(d)), target_count)
    else:
        idx = [rng.randint(0, len(d)-1) for _ in range(target_count)]
    sampled.extend([{"text": d[i]["text"]} for i in idx])
log(f"[data] weighted Thai sample size: {len(sampled)}")

thai_ds = Dataset.from_list(sampled)

# ---- English + math + code mix (Typhoon-2 30:70 recipe) -------------------
# Add these to push general capability and prevent catastrophic forgetting.
if MIX_EN:
    log("[data] mixing in English/math/code data (Typhoon-2 recipe)")
    en_recs = []
    target_en_per_source = max(2000, len(sampled) // 5)

    def render_messages(msgs):
        try:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        except Exception:
            return None

    def add_from(stream_iter, transform, limit):
        n = 0
        for r in stream_iter:
            if n >= limit: break
            try:
                msgs = transform(r)
            except Exception:
                continue
            if not msgs: continue
            text = render_messages(msgs)
            if text and 50 < len(text) < 8000:
                en_recs.append({"text": text})
                n += 1
        log(f"[data]   added {n} records")

    sources = [
        ("meta-math/MetaMathQA", "default", "train",
         lambda r: [{"role":"user","content": r["query"]},
                    {"role":"assistant","content": r["response"]}]),
        ("OpenCoder-LLM/opc-sft-stage1", "filtered_infinity_instruct", "train",
         lambda r: [{"role":"user","content": r.get("instruction") or r.get("input","")},
                    {"role":"assistant","content": r.get("output","")}]),
        ("teknium/OpenHermes-2.5", None, "train",
         lambda r: r.get("conversations") and [
             {"role":"user" if t["from"] in ("human","user") else "assistant",
              "content": t["value"]} for t in r["conversations"]
             if t["from"] in ("human","user","gpt","assistant")]),
    ]
    for spec in sources:
        name, config, split, fn = spec
        try:
            log(f"[data] streaming {name}/{config or ''} ...")
            args = (name,) if config is None else (name, config)
            ds_stream = load_dataset(*args, split=split, streaming=True)
            add_from(ds_stream, fn, target_en_per_source)
        except Exception as e:
            log(f"[data]   skipped {name}: {e}")

    log(f"[data] english mix size: {len(en_recs)}")
    en_ds = Dataset.from_list(en_recs)
    train_ds = concatenate_datasets([thai_ds, en_ds]).shuffle(seed=SEED)
else:
    train_ds = thai_ds.shuffle(seed=SEED)
log(f"[data] FINAL train size: {len(train_ds)}")

# ---- Model + LoRA ---------------------------------------------------------
log(f"[model] loading base (qlora={USE_QLORA})")
device_map = None if WORLD_SIZE > 1 else "auto"

if USE_QLORA:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    # For QLoRA in DDP, each rank must load to its own GPU
    if WORLD_SIZE > 1:
        device_map = {"": LOCAL_RANK}
    model = AutoModelForCausalLM.from_pretrained(
        BASE, quantization_config=bnb_config, device_map=device_map,
        trust_remote_code=True, attn_implementation="sdpa",
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
else:
    model = AutoModelForCausalLM.from_pretrained(
        BASE, torch_dtype=torch.bfloat16, device_map=device_map,
        trust_remote_code=True, attn_implementation="sdpa",
    )
model.config.use_cache = False

# Wide target_modules for stronger transfer
lora = LoraConfig(
    r=LORA_R, lora_alpha=LORA_R * 2, lora_dropout=0.05,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    bias="none", task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora)
if not hasattr(model.base_model.model, "warnings_issued"):
    model.base_model.model.warnings_issued = {}
if is_main: model.print_trainable_parameters()

# ---- Trainer config — interruptible-friendly ------------------------------
OUT.mkdir(parents=True, exist_ok=True)

# Auto-detect existing checkpoints for resume
def find_last_checkpoint(d):
    if not d.exists(): return None
    ckpts = sorted(d.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
    return str(ckpts[-1]) if ckpts else None

resume_from = find_last_checkpoint(OUT)
log(f"[ckpt] resume_from={resume_from}")

cfg = SFTConfig(
    output_dir=str(OUT),
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,   # eff batch 16 across 2 GPUs
    learning_rate=1e-5,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    logging_steps=20,
    save_steps=200,                   # checkpoint every 200 steps (~20 min) for preemption recovery
    save_total_limit=3,
    bf16=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    optim=("paged_adamw_8bit" if USE_QLORA else "adamw_torch_fused"),
    max_length=2048,
    packing=False,
    dataset_text_field="text",
    report_to="none",
    seed=SEED,
    dataloader_num_workers=2,
    ddp_find_unused_parameters=False,
)

trainer = SFTTrainer(model=model, args=cfg, train_dataset=train_ds, processing_class=tok)
log("[train] starting")
trainer.train(resume_from_checkpoint=resume_from)
trainer.save_model(str(OUT / "final"))
if is_main:
    tok.save_pretrained(str(OUT / "final"))
log("[train] DONE")
