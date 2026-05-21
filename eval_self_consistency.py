"""Self-consistency eval for AIME / MATH / OpenThaiEval / ThaiExam.

Sample n responses at temperature 0.7, take majority vote of extracted answers.
Typical gain: +5-10pp on math/MCQ. Free at inference time.

Usage:
    python3 eval_self_consistency.py \\
        --model Jnx03/kanitakorn-thaillm-8b-sft \\
        --benchmark aime24 \\
        --n 5 \\
        --out reports/aime24_sc.json
"""
import os, json, re, argparse, collections
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from datasets import load_dataset

# ---- Answer extractors ----------------------------------------------------
def extract_boxed(text: str):
    """Extract \\boxed{...} content. Returns last boxed value or None."""
    matches = re.findall(r"\\boxed\{([^{}]+)\}", text)
    return matches[-1].strip() if matches else None

def extract_after_marker(text: str, markers=("คำตอบคือ", "Answer:", "ANS:", "final answer is")):
    for m in markers:
        idx = text.rfind(m)
        if idx >= 0:
            tail = text[idx + len(m):].strip()
            # Take first number/expression
            mt = re.match(r"[^\d\-+]*([\-+]?\d+(?:\.\d+)?(?:/\d+)?|\([a-eA-E]\)|[a-eA-E])", tail)
            if mt: return mt.group(1).strip("()")
    return None

def extract_mcq_letter(text: str):
    # Find last (a)/(b)/.../(e) or "answer is a"
    matches = re.findall(r"[(\[]([a-eA-E])[)\]]", text)
    if matches: return matches[-1].lower()
    return None

def normalize_answer(s):
    if s is None: return None
    s = str(s).strip().lower()
    # Strip $, latex, trailing periods
    s = s.replace("$", "").replace("\\\\", "").rstrip(".")
    # Normalize fractions: "1/2" stays "1/2"
    return s

def extract(text: str, kind: str):
    """kind in {math, mcq, integer}"""
    if kind == "mcq":
        return normalize_answer(extract_mcq_letter(text) or extract_after_marker(text))
    # math/integer: prefer \boxed, then "Answer:"
    return normalize_answer(extract_boxed(text) or extract_after_marker(text))

# ---- Benchmarks -----------------------------------------------------------
def load_aime24():
    ds = load_dataset("math-ai/aime24", split="test")
    return [(r["problem"], str(r["answer"]).strip(), "integer") for r in ds]

def load_math500():
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    return [(r["problem"], normalize_answer(extract_boxed(r["solution"]) or str(r.get("answer",""))), "math") for r in ds]

def load_thaiexam():
    """scb10x/thai_exam has 5 configs: onet, ic, tgat, tpat1, a_level. Combine all."""
    configs = ["onet", "ic", "tgat", "tpat1", "a_level"]
    out = []
    for cfg in configs:
        try:
            ds = load_dataset("scb10x/thai_exam", cfg, split="test")
        except Exception as e:
            print(f"[thaiexam] skip {cfg}: {e}")
            continue
        for r in ds:
            q = r.get("question") or r.get("instruction") or ""
            # try multiple choice field naming schemes
            choices = []
            for i in "abcdeABCDE":
                v = r.get(f"choice_{i.lower()}") or r.get(f"option_{i.lower()}") or r.get(i.lower())
                if v: choices.append(v)
            if not choices:
                # Some configs have choices as list
                choices = r.get("choices") or r.get("options") or []
            if not q or not choices: continue
            prompt = q + "\n\nตัวเลือก:\n" + "\n".join(f"({chr(97+i)}) {c}" for i,c in enumerate(choices))
            gold = str(r.get("answer") or r.get("label") or "").strip().lower()
            if gold.isdigit(): gold = chr(96 + int(gold))  # 1→a, 2→b
            out.append((prompt, gold, "mcq"))
    return out

def load_openthaieval():
    """iapp/openthaieval — Thai O-NET / A-Level / TGAT / TPAT MCQ."""
    try:
        ds = load_dataset("iapp/openthaieval", split="test")
    except Exception:
        ds = load_dataset("iapp/openthaieval", split="train")
    out = []
    for r in ds:
        q = r.get("question") or r.get("instruction") or ""
        opts = []
        for k in ["a","b","c","d","e","A","B","C","D","E"]:
            v = r.get(k) or r.get(f"choice_{k}") or r.get(f"option_{k}")
            if v: opts.append(v)
        if not opts:
            opts = r.get("choices") or r.get("options") or []
        if not q or not opts: continue
        prompt = q + "\n\nตัวเลือก:\n" + "\n".join(f"({chr(97+i)}) {c}" for i,c in enumerate(opts))
        gold = (r.get("answer") or r.get("label") or "").strip().lower()
        # Normalize numeric to letter if needed
        if gold.isdigit(): gold = chr(96 + int(gold))
        out.append((prompt, gold, "mcq"))
    return out

def load_ifeval_th():
    """typhoon-ai/ifeval-th — Thai instruction following with verifiable constraints."""
    ds = load_dataset("typhoon-ai/ifeval-th", split="test")
    return [(r["prompt"], r.get("instruction_id_list") or r.get("instructions") or [], "ifeval") for r in ds]

def load_mt_bench_th():
    """ThaiLLM-Leaderboard/mt-bench-thai — MT-Bench Thai. Score via LLM judge."""
    ds = load_dataset("ThaiLLM-Leaderboard/mt-bench-thai", split="test")
    return [(r["turns"][0] if isinstance(r.get("turns"), list) else r.get("question_1", ""),
             r.get("category", "unknown"), "open_quality") for r in ds]

def load_aime25():
    ds = load_dataset("math-ai/aime25", split="test")
    return [(r["problem"], str(r["answer"]).strip(), "integer") for r in ds]

BENCHMARKS = {
    "aime24": load_aime24,
    "aime25": load_aime25,
    "math500": load_math500,
    "thaiexam": load_thaiexam,
    "openthaieval": load_openthaieval,
    "ifeval_th": load_ifeval_th,
    "mt_bench_th": load_mt_bench_th,
}

# ---- Self-consistency runner ---------------------------------------------
def run(args):
    print(f"[load] base={args.base} adapter={args.adapter}")
    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    if not getattr(tok, "chat_template", None):
        qwen_tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", trust_remote_code=True)
        tok.chat_template = qwen_tok.chat_template

    model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
        model = model.merge_and_unload()
    model.eval()

    items = BENCHMARKS[args.benchmark]()
    if args.limit: items = items[:args.limit]
    print(f"[eval] {args.benchmark}: {len(items)} items, n={args.n} samples each")

    results = []
    correct = 0
    for i, (prompt, gold, kind) in enumerate(items):
        msgs = [{"role":"user","content": prompt + "\n\nThink step by step. End with \\boxed{answer} or 'คำตอบคือ X'."}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = tok([text]*args.n, return_tensors="pt", truncation=True, max_length=1500, padding=True).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **enc, max_new_tokens=8192,
                do_sample=True, temperature=0.7, top_p=0.95,
                pad_token_id=tok.pad_token_id,
            )
        decoded = tok.batch_decode(outputs[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
        extracted = [extract(d, kind) for d in decoded]
        votes = collections.Counter(e for e in extracted if e is not None)
        pred = votes.most_common(1)[0][0] if votes else None
        gold_norm = normalize_answer(gold)
        is_correct = (pred == gold_norm)
        correct += int(is_correct)
        results.append({
            "idx": i, "gold": gold_norm, "pred": pred,
            "votes": dict(votes), "samples": decoded,
            "correct": is_correct,
        })
        if (i+1) % 10 == 0:
            print(f"  [{i+1}/{len(items)}] running acc = {correct/(i+1):.3f}")

    acc = correct / len(items)
    print(f"\n[final] {args.benchmark} self-consistency (n={args.n}): {acc:.4f}")

    out = {
        "benchmark": args.benchmark, "n_samples": args.n,
        "n_items": len(items), "correct": correct, "accuracy": acc,
        "base": args.base, "adapter": args.adapter,
        "items": results,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[out] saved to {args.out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3-8B-Base")
    ap.add_argument("--adapter", default=None, help="path to LoRA adapter (optional)")
    ap.add_argument("--benchmark", choices=list(BENCHMARKS), required=True)
    ap.add_argument("--n", type=int, default=5, help="samples for self-consistency")
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--out", required=True)
    run(ap.parse_args())
