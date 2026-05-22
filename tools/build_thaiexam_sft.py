"""Convert our targeted Thai exam MCQs to SFT records with high-quality CoT.

Input: train_thaiexam_targeted*.jsonl with {question, choices, answer, explanation}
Output: dataset/sft_ready_thai_v5/thaiexam_targeted_train.jsonl

Format aligns with eval_vllm.py thaiexam prompt — each record:
  {messages: [{role:user, content: "Q + choices"},
              {role:assistant, content: "<reasoning> คำตอบคือ (X)"}]}

The assistant response uses the explanation as reasoning, ending with the
exact answer marker our eval expects.
"""
import argparse, json, re
from pathlib import Path

def render(rec):
    q = rec.get("question", "").strip()
    choices = rec.get("choices", [])
    ans = str(rec.get("answer", "")).strip()
    expl = rec.get("explanation", "").strip()
    if not q or len(choices) != 5 or ans not in {"1","2","3","4","5"} or not expl:
        return None
    # Convert numeric answer 1-5 to letter a-e
    letter = chr(ord('a') + int(ans) - 1)
    # Build user prompt matching our eval style
    user = q + "\n\nตัวเลือก:\n" + "\n".join(f"({chr(97+i)}) {c}" for i, c in enumerate(choices))
    suffix = "\n\nคิดและให้เหตุผลอย่างละเอียด แล้วระบุคำตอบในบรรทัดสุดท้ายในรูปแบบ:\nคำตอบคือ (X)\n\nโดย X ต้องเป็นตัวอักษรจากตัวเลือกข้างต้นเท่านั้น (a, b, c, d, หรือ e — ห้ามใช้ตัวอักษรอื่น)"
    user += suffix
    # Assistant: explanation + correct answer marker
    assistant = expl
    if "คำตอบคือ" not in assistant:
        assistant += f"\n\nคำตอบคือ ({letter})"
    return {"messages": [
        {"role":"user","content": user},
        {"role":"assistant","content": assistant},
    ], "category": rec.get("category", "thai")}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n_in = 0; n_out = 0
    seen_q = set()
    with out.open("w", encoding="utf-8") as fh:
        for src in args.inputs:
            for line in Path(src).open(encoding="utf-8"):
                line = line.strip()
                if not line: continue
                try:
                    rec = json.loads(line)
                except: continue
                n_in += 1
                q = rec.get("question","")[:200]
                if q in seen_q: continue
                seen_q.add(q)
                rendered = render(rec)
                if rendered:
                    fh.write(json.dumps(rendered, ensure_ascii=False) + "\n")
                    n_out += 1
    print(f"[done] {n_in} input → {n_out} unique SFT records -> {args.out}")

if __name__ == "__main__":
    main()
