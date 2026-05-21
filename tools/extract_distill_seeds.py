"""Extract (problem, answer) pairs from existing SFT records to use as
distillation seeds. Outputs a JSONL ready for tools/distill_from_teacher.py.

Reads dataset/train/train_aime_th_*.jsonl, parses the chat-template `text`
or `messages` field, pulls the last 'Your task' problem and the assistant's
boxed answer.
"""
import json, re, glob, argparse
from pathlib import Path

def extract_from_record(rec):
    """Returns (problem_text, answer) or None."""
    text = rec.get("text", "")
    if not text and "messages" in rec:
        # multi-turn — last user is the actual problem
        for m in reversed(rec["messages"]):
            if m["role"] == "user":
                return m["content"].strip(), None
        return None
    # Chat-template format
    if "# Your task" in text:
        prob = text.split("# Your task")[-1]
        prob = prob.split("<|im_end|>")[0].split("<|im_start|>")[0].strip()
        # Pull answer from the assistant turn
        ans = None
        if "<|im_start|>assistant" in text:
            asst = text.split("<|im_start|>assistant")[-1]
            asst = asst.split("<|im_end|>")[0]
            # Look for boxed{...} or "คำตอบคือ X"
            m = re.search(r"\boxed\{([^{}]+)\}", asst)
            if m: ans = m.group(1).strip()
            if not ans:
                m = re.search(r"คำตอบคือ\s*([0-9\-\+]+)", asst)
                if m: ans = m.group(1).strip()
        return prob, ans
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-glob", default="dataset/train/train_aime_th_*.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max", type=int, default=2000)
    ap.add_argument("--language", default="th")
    args = ap.parse_args()

    out_recs = []
    seen = set()
    for fp in glob.glob(args.input_glob):
        for line in Path(fp).open(encoding="utf-8"):
            if not line.strip(): continue
            try: r = json.loads(line)
            except: continue
            res = extract_from_record(r)
            if not res: continue
            prob, ans = res
            if not prob or len(prob) < 20: continue
            key = prob[:200]
            if key in seen: continue
            seen.add(key)
            out_recs.append({
                "id": r.get("id"),
                "question": prob,
                "answer": ans,
                "language": args.language,
                "source": Path(fp).name,
            })
            if len(out_recs) >= args.max: break
        if len(out_recs) >= args.max: break

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.out).open("w", encoding="utf-8") as fh:
        for r in out_recs:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[extracted] {len(out_recs)} seeds -> {args.out}")

if __name__ == "__main__":
    main()
