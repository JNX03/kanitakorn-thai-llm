"""Convert downloaded Typhoon/Wangchan datasets to our SFT format.

Output: dataset/sft_ready_external/*.jsonl
Each line: {"messages": [{"role":"user", "content":"..."}, {"role":"assistant", "content":"..."}]}
"""
import json, sys
from pathlib import Path

def has_thai(text, min_ratio=0.3, scan_chars=500):
    if not text: return False
    s = text[:scan_chars]
    return sum(1 for c in s if '฀' <= c <= '๿') / max(len(s), 1) >= min_ratio

def convert_typhoon_t1_thai(in_path, out_path):
    """Typhoon T1 structured_thai — Thai reasoning traces."""
    n_in = n_out = 0
    with Path(in_path).open(encoding="utf-8") as fp, Path(out_path).open("w", encoding="utf-8") as out:
        for line in fp:
            n_in += 1
            try: r = json.loads(line)
            except: continue
            # Format vary — try common fields
            q = r.get("question") or r.get("prompt") or r.get("instruction") or ""
            a = r.get("response") or r.get("output") or r.get("answer") or ""
            reasoning = r.get("reasoning") or r.get("trace") or r.get("thought") or ""
            if reasoning and a:
                full_response = f"{reasoning}\n\n{a}" if reasoning != a else a
            else:
                full_response = a or reasoning
            if not q or not full_response: continue
            if not (has_thai(q) or has_thai(full_response)): continue
            out.write(json.dumps({"messages": [
                {"role":"user","content": q},
                {"role":"assistant","content": full_response},
            ]}, ensure_ascii=False) + "\n")
            n_out += 1
    print(f"  {in_path} -> {n_out}/{n_in}")
    return n_out

def convert_typhoon_s(in_path, out_path, limit=20000):
    n_in = n_out = 0
    with Path(in_path).open(encoding="utf-8") as fp, Path(out_path).open("w", encoding="utf-8") as out:
        for line in fp:
            if n_out >= limit: break
            n_in += 1
            try: r = json.loads(line)
            except: continue
            msgs = r.get("messages") or r.get("conversations") or []
            # Convert role names
            if msgs and isinstance(msgs, list):
                fmt = []
                for m in msgs:
                    role = m.get("role") or m.get("from", "")
                    content = m.get("content") or m.get("value", "")
                    if role in ("human","user"): role = "user"
                    elif role in ("gpt","assistant","bot"): role = "assistant"
                    if role in ("user","assistant") and content:
                        fmt.append({"role":role,"content":content})
                if len(fmt) >= 2:
                    # Need Thai in at least one message
                    if any(has_thai(m["content"]) for m in fmt):
                        out.write(json.dumps({"messages": fmt}, ensure_ascii=False) + "\n")
                        n_out += 1
                        continue
            # Fallback: prompt/response style
            q = r.get("prompt") or r.get("instruction") or r.get("question") or ""
            a = r.get("response") or r.get("output") or r.get("answer") or ""
            if not q or not a: continue
            if not (has_thai(q) or has_thai(a)): continue
            out.write(json.dumps({"messages":[{"role":"user","content":q},{"role":"assistant","content":a}]}, ensure_ascii=False) + "\n")
            n_out += 1
    print(f"  {in_path} -> {n_out}/{n_in} (limit {limit})")
    return n_out

def convert_wangchan(in_path, out_path, limit=20000):
    n_in = n_out = 0
    with Path(in_path).open(encoding="utf-8") as fp, Path(out_path).open("w", encoding="utf-8") as out:
        for line in fp:
            if n_out >= limit: break
            n_in += 1
            try: r = json.loads(line)
            except: continue
            # Wangchan uses capitalized keys
            q = (r.get("Instruction") or r.get("instruction") or "").strip()
            inp = (r.get("Input") or r.get("input") or "").strip()
            if inp: q = q + "\n" + inp
            a = (r.get("Output") or r.get("output") or r.get("response") or "").strip()
            if not q or not a: continue
            if not (has_thai(q) or has_thai(a)): continue
            out.write(json.dumps({"messages":[{"role":"user","content":q},{"role":"assistant","content":a}]}, ensure_ascii=False) + "\n")
            n_out += 1
    print(f"  {in_path} -> {n_out}/{n_in} (limit {limit})")
    return n_out

def main():
    Path("dataset/sft_ready_external").mkdir(parents=True, exist_ok=True)
    total = 0
    # Typhoon T1 Thai (reasoning)
    total += convert_typhoon_t1_thai(
        "dataset/external/scb10x_typhoon-t1-3b-sci-fm-iclr-2025-exp-dataset_train_structured_thai.jsonl",
        "dataset/sft_ready_external/typhoon_t1_thai_train.jsonl"
    )
    # Typhoon T1 EN (limit 5000 for diversity)
    total += convert_typhoon_t1_thai(
        "dataset/external/scb10x_typhoon-t1-3b-sci-fm-iclr-2025-exp-dataset_train_structured.jsonl",
        "dataset/sft_ready_external/typhoon_t1_en_train.jsonl"
    )
    # Typhoon S SFT (limit 15K for Thai content)
    total += convert_typhoon_s(
        "dataset/external/typhoon-ai_typhoon-s-instruct-post-training_sft.jsonl",
        "dataset/sft_ready_external/typhoon_s_sft_train.jsonl",
        limit=15000
    )
    # Wangchan
    total += convert_wangchan(
        "dataset/external/airesearch_WangchanThaiInstruct_train.jsonl",
        "dataset/sft_ready_external/wangchan_train.jsonl",
        limit=15000
    )
    print(f"\nTOTAL: {total} SFT records from external datasets")

if __name__ == "__main__":
    main()
