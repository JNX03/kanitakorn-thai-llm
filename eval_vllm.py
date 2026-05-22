"""vLLM-based eval — 5-10x faster than HF generate() for batched inference.

For 14B model on 2x A100 40GB: ~150 tokens/sec aggregate, ~20s per record
with 4096 max tokens. 100 records = ~30 min vs ~60 min with HF.

Usage:
    python3 eval_vllm.py \\
        --base deepseek-ai/DeepSeek-R1-Distill-Qwen-14B \\
        --benchmark math500 \\
        --limit 100 \\
        --out reports/base_vllm_math500.json
"""
import os, json, argparse, re
from pathlib import Path

# Re-use loaders from eval_self_consistency
from eval_self_consistency import BENCHMARKS, extract, normalize_answer

def run(args):
    print(f"[vllm] base={args.base} adapter={args.adapter} bench={args.benchmark}")
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.base,
        tensor_parallel_size=args.tp,
        max_model_len=args.max_model_len,
        dtype="bfloat16",
        enable_lora=bool(args.adapter),
        max_loras=1, max_lora_rank=64,
        gpu_memory_utilization=0.92,
    )
    lora_request = None
    if args.adapter:
        from vllm.lora.request import LoRARequest
        lora_request = LoRARequest("adapter", 1, args.adapter)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    if not getattr(tok, "chat_template", None):
        qwen_tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", trust_remote_code=True)
        tok.chat_template = qwen_tok.chat_template

    items = BENCHMARKS[args.benchmark]()
    if args.limit: items = items[:args.limit]
    print(f"[vllm] {args.benchmark}: {len(items)} items, n={args.n}, max_tokens={args.max_tokens}")

    prompts = []
    for prompt, gold, kind in items:
        if kind == "mcq":
            # Thai-language prompt with strict format
            suffix = "\n\nคิดและให้เหตุผลอย่างละเอียด แล้วระบุคำตอบในบรรทัดสุดท้ายในรูปแบบ:\nคำตอบคือ (X)\n\nโดย X ต้องเป็นตัวอักษรจากตัวเลือกข้างต้นเท่านั้น (a, b, c, d, หรือ e — ห้ามใช้ตัวอักษรอื่น)"
        else:
            suffix = "\n\nThink step by step. End with \\boxed{answer}."
        msgs = [{"role":"user","content": prompt + suffix}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        prompts.append(text)

    sampling = SamplingParams(
        n=args.n, temperature=(0.6 if args.n > 1 else 0.0),
        top_p=0.95, max_tokens=args.max_tokens,
    )

    print("[vllm] generating...")
    outputs = llm.generate(prompts, sampling, lora_request=lora_request)

    correct = 0
    records = []
    for i, (out, (prompt, gold, kind)) in enumerate(zip(outputs, items)):
        gold_n = normalize_answer(gold)
        samples = [o.text for o in out.outputs]
        extracted = [extract(s, kind) for s in samples]
        # Majority vote across n samples
        from collections import Counter
        votes = Counter(e for e in extracted if e is not None)
        pred = votes.most_common(1)[0][0] if votes else None
        ok = (pred == gold_n)
        correct += int(ok)
        records.append({
            "idx": i, "gold": gold_n, "pred": pred,
            "votes": dict(votes), "n_samples": len(samples),
            "samples": samples[:2],  # save first 2 samples for debugging
            "correct": ok,
        })

    acc = correct / len(items)
    print(f"\n[final] {args.benchmark} vLLM n={args.n}: {acc:.4f} ({correct}/{len(items)})")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "benchmark": args.benchmark, "n_samples": args.n,
        "n_items": len(items), "correct": correct, "accuracy": acc,
        "base": args.base, "adapter": args.adapter,
        "max_tokens": args.max_tokens, "method": "vllm",
        "items": records,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[out] {args.out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--benchmark", choices=list(BENCHMARKS), required=True)
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--tp", type=int, default=2, help="tensor parallel (use 2 for 2x A100)")
    ap.add_argument("--out", required=True)
    run(ap.parse_args())
