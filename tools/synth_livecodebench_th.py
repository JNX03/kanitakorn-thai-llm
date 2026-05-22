"""Synthesize LiveCodeBench-style coding problems in Thai for SFT.

Format aligned with eval expectation:
  {messages: [{role:user, content: "Python problem in Thai"},
              {role:assistant, content: "<reasoning> ```python\n<code>\n```"}]}
"""
import argparse, json, subprocess, threading, time, shutil, os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = Path(__file__).resolve().parents[1]
CODEX = (shutil.which("codex") or shutil.which("codex.cmd")
         or os.path.expanduser("~/AppData/Roaming/npm/codex.cmd"))

PROMPT = """สร้างโจทย์ programming 5 ข้อ ระดับยาก (LeetCode medium-hard, contest-style)

แต่ละข้อต้องมี:
- คำอธิบายปัญหาเป็นภาษาไทย ชัดเจน รวม input/output format และ constraints
- ตัวอย่าง input/output 2-3 ตัวอย่าง
- โซลูชัน Python ที่ทำงานได้จริง (ฟังก์ชันรับ input, return output)
- คำอธิบาย algorithm สั้นๆ เป็นภาษาไทย

หัวข้อหลากหลาย: arrays, strings, dynamic programming, graphs, trees, recursion, math.

ส่งคืน JSON:
{
  "items": [
    {"problem": "...", "solution": "def solve(...): ...", "explanation": "..."},
    ...
  ]
}"""

SCHEMA = json.dumps({
    "type":"object","additionalProperties":False,
    "properties":{"items":{"type":"array","items":{
        "type":"object","additionalProperties":False,
        "properties":{
            "problem":{"type":"string"},"solution":{"type":"string"},
            "explanation":{"type":"string"}
        },"required":["problem","solution","explanation"]
    }}},"required":["items"]
})

LOCK = threading.Lock()
STATS = {"got":0}

def call_codex(prompt: str) -> list:
    if not CODEX: return []
    schema_path = ROOT / "synth_lcb_schema.json"
    schema_path.write_text(SCHEMA, encoding="utf-8")
    args = [CODEX,"exec","--skip-git-repo-check","--ephemeral","-s","read-only",
            "--output-schema",str(schema_path),"--color","never",
            "-c","model_reasoning_effort=\"low\"", prompt]
    try:
        proc = subprocess.run(args, capture_output=True, text=True, encoding='utf-8',
                              errors='replace', timeout=300)
        text = (proc.stdout or "") + (proc.stderr or "")
    except subprocess.TimeoutExpired: return []
    i = text.find('{"items"')
    if i < 0: return []
    depth = 0; end = i
    for j in range(i, len(text)):
        if text[j] == '{': depth += 1
        elif text[j] == '}':
            depth -= 1
            if depth == 0: end = j+1; break
    try:
        return json.loads(text[i:end]).get("items", [])
    except: return []

def filter_valid(items):
    out = []
    for it in items:
        p = (it.get("problem") or "").strip()
        s = (it.get("solution") or "").strip()
        e = (it.get("explanation") or "").strip()
        if not p or not s or not e: continue
        # Need Thai
        if not any('฀' <= c <= '๿' for c in p[:1000]): continue
        # Solution must be Python with def
        if "def " not in s: continue
        out.append({"problem": p, "solution": s, "explanation": e})
    return out

def worker(fh):
    items = call_codex(PROMPT)
    valid = filter_valid(items)
    with LOCK:
        for it in valid:
            # Convert to SFT format
            user = it["problem"] + "\n\nเขียนโซลูชัน Python ที่ถูกต้อง"
            assistant = f"{it['explanation']}\n\n```python\n{it['solution']}\n```"
            rec = {"messages": [
                {"role":"user","content": user},
                {"role":"assistant","content": assistant},
            ]}
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
        STATS["got"] += len(valid)
        print(f"  +{len(valid)} (total: {STATS['got']})")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=300)
    ap.add_argument("--concurrency", type=int, default=5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fh = Path(args.out).open("w", encoding="utf-8")
    n_calls = (args.target // 4) + 5
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(worker, fh) for _ in range(n_calls)]
        for f in as_completed(futs):
            f.result()
            if STATS["got"] >= args.target: break
    fh.close()
    print(f"\n[done] {STATS['got']} records in {time.time()-t0:.0f}s -> {args.out}")

if __name__ == "__main__":
    main()
