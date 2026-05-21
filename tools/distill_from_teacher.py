"""Distill R1-style CoT traces from a strong teacher via OpenRouter.

Generates {prompt → R1-style CoT solution} records for our training corpus.
Targets math preservation during Thai SFT — if we generate Thai R1 traces
on AIME-style problems, the student keeps math reasoning AND learns to
output it in Thai.

Teachers (rank-ordered by AIME24 perf, cheapest first):
  - deepseek/deepseek-r1                       (cheapest R1, ~$0.50/M out)
  - deepseek/deepseek-r1-distill-llama-70b     (distillation budget tier)
  - anthropic/claude-3.5-sonnet                (fallback if R1 rate-limited)

Usage:
    python3 tools/distill_from_teacher.py \\
        --problems-jsonl seeds/aime_problems_th.jsonl \\
        --out dataset/train/train_distill_aime_th_000.jsonl \\
        --teacher deepseek/deepseek-r1 \\
        --n 200 --concurrency 4 --max-tokens 4096
"""
import argparse, json, os, time, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

OR_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OR_URL = "https://openrouter.ai/api/v1/chat/completions"

LOCK = threading.Lock()
STATS = {"sent": 0, "got": 0, "failed": 0}

def call_teacher(problem: str, teacher: str, max_tokens: int, lang: str = "th"):
    """Single OpenRouter call. Returns (cot_text, raw_response) or (None, err)."""
    sys_msg = {
        "th": "คุณเป็นนักคณิตศาสตร์ระดับโอลิมปิก แก้ปัญหาต่อไปนี้แบบ chain-of-thought เป็นภาษาไทย ขึ้นต้นด้วย <think> ... </think> แล้วตามด้วยคำตอบสุดท้ายในรูป \\boxed{ANSWER}",
        "en": "You are an olympiad mathematician. Solve the following problem step by step, wrapping reasoning in <think>...</think>, ending with \\boxed{ANSWER}.",
    }[lang]
    body = {
        "model": teacher,
        "messages": [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": problem},
        ],
        "temperature": 0.6,
        "max_tokens": max_tokens,
    }
    headers = {"Authorization": f"Bearer {OR_KEY}", "Content-Type": "application/json"}
    try:
        r = requests.post(OR_URL, headers=headers, json=body, timeout=180)
        r.raise_for_status()
        j = r.json()
        text = j["choices"][0]["message"]["content"]
        return text, j
    except requests.exceptions.HTTPError as e:
        return None, f"HTTP {r.status_code}: {r.text[:300]}"
    except Exception as e:
        return None, str(e)

def worker(rec, teacher, max_tokens, out_fh):
    problem = rec["question"] if "question" in rec else rec.get("problem") or rec.get("prompt")
    if not problem: return
    lang = rec.get("language", "th")
    with LOCK: STATS["sent"] += 1
    text, raw = call_teacher(problem, teacher, max_tokens, lang)
    if text is None:
        with LOCK:
            STATS["failed"] += 1
            print(f"[fail {STATS['failed']}] {raw[:200]}")
        return
    out = {
        "id": rec.get("id"),
        "language": lang,
        "messages": [
            {"role": "user", "content": problem},
            {"role": "assistant", "content": text},
        ],
        "teacher": teacher,
        "source": rec.get("source", "distill_unknown"),
        "gold": rec.get("answer") or rec.get("gold"),
    }
    with LOCK:
        out_fh.write(json.dumps(out, ensure_ascii=False) + "\n")
        out_fh.flush()
        STATS["got"] += 1
        if STATS["got"] % 5 == 0:
            print(f"  [{STATS['got']}/{STATS['sent']}] (failed={STATS['failed']})")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems-jsonl", required=True, help="JSONL of problems with {id, question/problem, answer, language}")
    ap.add_argument("--out", required=True)
    ap.add_argument("--teacher", default="deepseek/deepseek-r1")
    ap.add_argument("--n", type=int, default=200, help="Max problems to distill")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=4096)
    args = ap.parse_args()

    if not OR_KEY: raise SystemExit("OPENROUTER_API_KEY not set")
    problems = []
    for line in Path(args.problems_jsonl).open(encoding="utf-8"):
        if not line.strip(): continue
        problems.append(json.loads(line))
        if len(problems) >= args.n: break
    print(f"[distill] {len(problems)} problems → {args.teacher}, concurrency={args.concurrency}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out_fh = Path(args.out).open("w", encoding="utf-8")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(worker, p, args.teacher, args.max_tokens, out_fh) for p in problems]
        for f in as_completed(futs):
            f.result()
    out_fh.close()
    elapsed = time.time() - t0
    print(f"\n[done] {STATS['got']}/{STATS['sent']} traces in {elapsed:.1f}s (failed={STATS['failed']})")
    print(f"[out] {args.out}")

if __name__ == "__main__":
    main()
