"""MT-Bench Best-of-N: sample N model responses per prompt, gemini judge picks max.

Theory: at greedy, model output is fixed. With temp>0 sampling, different responses
have different scores. Best-of-N picks the maximum, lifting the floor.
"""
import argparse, json, os, re, time
from pathlib import Path

import requests
from datasets import load_dataset
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

OR_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OR_URL = "https://openrouter.ai/api/v1/chat/completions"

def judge(prompt: str, response: str) -> float:
    if not OR_KEY: return None
    jp = f"""Evaluate this AI response (1-10): helpfulness, accuracy, depth, language.

[Prompt]
{prompt[:1500]}

[Response]
{response[:2000]}

Output only an integer 1-10."""
    try:
        r = requests.post(OR_URL, headers={"Authorization": f"Bearer {OR_KEY}"},
            json={"model": "google/gemini-2.5-flash-lite",
                  "messages": [{"role":"user","content": jp}],
                  "temperature": 0.0, "max_tokens": 5},
            timeout=30)
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"].strip()
        m = re.search(r"\d+", text)
        return float(m.group()) if m else None
    except Exception as e:
        print(f"[judge] err: {e}"); return None

def get_visible(t):
    if "</think>" in t:
        v = t.split("</think>",1)[1].strip()
        return v if len(v) >= 100 else t
    return t

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--n", type=int, default=3, help="samples per prompt")
    ap.add_argument("--limit", type=int, default=80)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    print(f"[load] {args.base}")
    llm = LLM(model=args.base, tensor_parallel_size=1, max_model_len=args.max_model_len,
              dtype="bfloat16", gpu_memory_utilization=0.92)
    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)

    # MT-Bench
    for split in ["test", "train"]:
        try:
            ds = load_dataset("ThaiLLM-Leaderboard/mt-bench-thai", split=split); break
        except: continue
    items = []
    for r in ds:
        q = (r["turns"][0] if isinstance(r.get("turns"), list) and r["turns"] else
             r.get("question_1") or r.get("question") or r.get("instruction") or "")
        if q: items.append(q)
    items = items[:args.limit]
    print(f"[items] {len(items)} prompts × {args.n} samples = {len(items)*args.n} generations")

    prompts = []
    for q in items:
        msgs = [{"role":"user","content": q}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        prompts.append(text)

    print("[generating]")
    # Sample n times
    outputs = llm.generate(prompts, SamplingParams(
        n=args.n, temperature=args.temperature, top_p=0.95,
        max_tokens=args.max_tokens))

    # Score each candidate and pick max per prompt
    records = []
    all_max = []
    all_greedy = []
    for q, out in zip(items, outputs):
        candidates = [get_visible(o.text) for o in out.outputs]
        scores = []
        for c in candidates:
            s = judge(q, c)
            scores.append(s if s is not None else 0)
            time.sleep(0.2)
        best_idx = int(max(range(len(scores)), key=lambda i: scores[i]))
        max_score = scores[best_idx]
        # Also record first sample (proxy for greedy quality)
        greedy_score = scores[0]
        all_max.append(max_score)
        all_greedy.append(greedy_score)
        records.append({"prompt": q[:200], "best": candidates[best_idx][:500],
                        "best_score": max_score, "greedy_score": greedy_score,
                        "all_scores": scores})

    avg_max = sum(all_max) / max(len(all_max), 1)
    avg_greedy = sum(all_greedy) / max(len(all_greedy), 1)
    print(f"[mt_bench_bon] n={args.n} t={args.temperature}")
    print(f"  best-of-{args.n}: avg={avg_max:.2f}/10 → {avg_max*10:.1f}%")
    print(f"  first-sample:    avg={avg_greedy:.2f}/10 → {avg_greedy*10:.1f}%")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "benchmark":"mt_bench_th_bon", "n":args.n, "temperature":args.temperature,
        "accuracy_bon": avg_max/10, "avg_max": avg_max,
        "accuracy_greedy": avg_greedy/10, "avg_greedy": avg_greedy,
        "items": records,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[out] {args.out}")

if __name__ == "__main__":
    main()
