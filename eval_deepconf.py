"""DeepConf-style eval: confidence-weighted majority voting.

Research (arXiv DeepConf, 2025) shows +20.9pp over single-sample on AIME24
for DeepSeek-R1-Distill-Qwen-8B. We approximate the recipe:

  1. Sample n traces at temperature 0.6, top_p=0.95
  2. For each trace, compute mean token-level log-prob = "confidence"
  3. Extract answer from each trace
  4. Vote weighted by exp(mean_logprob) — high-confidence traces count more
  5. Optional: drop bottom-k confidence traces before voting (DeepConf "early termination")

Usage:
    python3 eval_deepconf.py \\
        --base deepseek-ai/DeepSeek-R1-Distill-Qwen-14B \\
        --adapter /root/kanitakorn/runs/.../final \\
        --benchmark aime24 \\
        --n 32 \\
        --drop-bottom-frac 0.25 \\
        --out reports/aime24_deepconf.json
"""
import argparse, json, os, re, collections, math
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from datasets import load_dataset

from eval_self_consistency import (
    extract, normalize_answer, BENCHMARKS,
)

def gen_with_logprobs(model, tok, prompt: str, n: int, max_new_tokens: int = 1024,
                     temperature: float = 0.6, top_p: float = 0.95):
    """Generate n samples; return list of (text, mean_logprob)."""
    msgs = [{"role":"user","content": prompt + "\n\nThink step by step. End with \\boxed{answer} or 'คำตอบคือ X'."}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    enc = tok(text, return_tensors="pt", truncation=True, max_length=1500).to(model.device)
    in_len = enc.input_ids.shape[1]

    # Expand to n
    input_ids = enc.input_ids.expand(n, -1)
    attn = enc.attention_mask.expand(n, -1)

    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids, attention_mask=attn,
            max_new_tokens=max_new_tokens,
            do_sample=True, temperature=temperature, top_p=top_p,
            pad_token_id=tok.pad_token_id,
            return_dict_in_generate=True, output_scores=True,
        )

    sequences = out.sequences[:, in_len:]
    # Compute mean log-prob per sequence
    # out.scores is a tuple of len max_new_tokens, each [n, vocab_size]
    results = []
    for i in range(n):
        token_logprobs = []
        for step, scores in enumerate(out.scores):
            if step >= sequences.shape[1]: break
            tok_id = sequences[i, step].item()
            if tok_id == tok.pad_token_id or tok_id == tok.eos_token_id: break
            logits = scores[i].float()
            # Softmax → log_softmax for numerical stability
            log_probs = torch.log_softmax(logits, dim=-1)
            token_logprobs.append(log_probs[tok_id].item())
        decoded = tok.decode(sequences[i], skip_special_tokens=True)
        mean_lp = sum(token_logprobs) / max(len(token_logprobs), 1)
        results.append((decoded, mean_lp))
    return results

def deepconf_vote(samples: list, kind: str, drop_bottom_frac: float = 0.25):
    """Confidence-weighted majority vote.
    samples: list of (text, mean_logprob)
    Returns (predicted_answer, votes_dict, kept_count)
    """
    # Extract answers + use exp(mean_lp) as weight
    extracted = [(extract(t, kind), lp) for t, lp in samples]
    # Sort by lp descending, drop bottom-k
    sorted_by_conf = sorted(extracted, key=lambda x: -x[1])
    keep = int(len(sorted_by_conf) * (1 - drop_bottom_frac))
    kept = sorted_by_conf[:max(keep, 1)]

    votes = collections.defaultdict(float)
    for ans, lp in kept:
        if ans is None: continue
        w = math.exp(lp)  # high-confidence → larger weight
        votes[ans] += w
    if not votes:
        return None, {}, len(kept)
    pred = max(votes.items(), key=lambda x: x[1])[0]
    return pred, dict(votes), len(kept)

def run(args):
    print(f"[load] base={args.base} adapter={args.adapter}")
    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    if not getattr(tok, "chat_template", None):
        qwen_tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", trust_remote_code=True)
        tok.chat_template = qwen_tok.chat_template

    model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
        model = model.merge_and_unload()
    model.eval()

    items = BENCHMARKS[args.benchmark]()
    if args.limit: items = items[:args.limit]
    print(f"[deepconf] benchmark={args.benchmark} n_items={len(items)} n_samples={args.n}")

    out_records = []
    correct = 0
    for i, (prompt, gold, kind) in enumerate(items):
        samples = gen_with_logprobs(model, tok, prompt, n=args.n,
                                     max_new_tokens=args.max_new_tokens)
        pred, votes, kept = deepconf_vote(samples, kind, args.drop_bottom_frac)
        gold_n = normalize_answer(gold)
        ok = (pred == gold_n)
        correct += int(ok)
        out_records.append({
            "idx": i, "gold": gold_n, "pred": pred, "votes": votes,
            "n_kept": kept, "correct": ok,
        })
        if (i+1) % 5 == 0:
            print(f"  [{i+1}/{len(items)}] running acc = {correct/(i+1):.3f}")

    acc = correct / len(items)
    print(f"\n[final] {args.benchmark} DeepConf (n={args.n}, drop={args.drop_bottom_frac}): {acc:.4f}")
    out = {
        "benchmark": args.benchmark, "method": "deepconf",
        "n_samples": args.n, "drop_bottom_frac": args.drop_bottom_frac,
        "n_items": len(items), "correct": correct, "accuracy": acc,
        "base": args.base, "adapter": args.adapter,
        "items": out_records,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[out] saved to {args.out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--benchmark", choices=list(BENCHMARKS), required=True)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--drop-bottom-frac", type=float, default=0.25)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    run(ap.parse_args())
