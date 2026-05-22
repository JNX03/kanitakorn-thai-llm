"""Synthesize hard math problems in Thai (AIME-style, MATH500-style).

Both problem in Thai AND solution in Thai. Helps Thai math capability.
"""
import argparse, json, subprocess, threading, time, shutil, os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = Path(__file__).resolve().parents[1]
CODEX = (shutil.which("codex") or shutil.which("codex.cmd")
         or os.path.expanduser("~/AppData/Roaming/npm/codex.cmd"))

LEVELS = [
    "AIME-style — Algebra, Number Theory, Combinatorics, Geometry. answer = positive integer 0-999",
    "MATH500-style — Algebra, Geometry, Probability, Calculus, Precalculus. answer = simplified",
    "Olympiad-style — IMO Day 1 style, proof or numeric",
]

PROMPT = """สร้างโจทย์คณิตศาสตร์ระดับยาก 5 ข้อ ในแนว: {level}

แต่ละข้อต้องมี:
- โจทย์เป็นภาษาไทย ระดับยากจริง (ห้ามใช้ตัวเลขเล็กๆ พื้นฐาน)
- คำตอบที่แน่นอน (เลขจำนวนเต็มสำหรับ AIME, หรือคำตอบในรูปแบบที่ลดทอนแล้ว)
- โซลูชันละเอียดเป็นภาษาไทย แสดงทุกขั้นตอน ด้วย LaTeX สำหรับสมการ

ส่งคืน JSON:
{{
  "items": [
    {{"problem": "...", "answer": "...", "solution": "..."}},
    ...
  ]
}}"""

SCHEMA = json.dumps({
    "type":"object","additionalProperties":False,
    "properties":{"items":{"type":"array","items":{
        "type":"object","additionalProperties":False,
        "properties":{
            "problem":{"type":"string"},"answer":{"type":"string"},"solution":{"type":"string"}
        },"required":["problem","answer","solution"]
    }}},"required":["items"]
})

LOCK = threading.Lock()
STATS = {"got":0}

def call_codex(prompt: str) -> list:
    if not CODEX: return []
    schema_path = ROOT / "synth_math_schema.json"
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

def worker(level, fh):
    items = call_codex(PROMPT.format(level=level))
    valid = []
    for it in items:
        p = (it.get("problem") or "").strip()
        a = (it.get("answer") or "").strip()
        s = (it.get("solution") or "").strip()
        if not p or not a or not s: continue
        if not any('฀' <= c <= '๿' for c in p[:500]): continue
        if not any('฀' <= c <= '๿' for c in s[:500]): continue
        # Format as SFT
        user = p + "\n\nคิดและให้เหตุผลอย่างละเอียด แล้วปิดท้ายด้วย \\boxed{คำตอบ}"
        assistant = s + f"\n\n\\boxed{{{a}}}"
        valid.append({"messages":[
            {"role":"user","content":user},
            {"role":"assistant","content":assistant},
        ]})
    with LOCK:
        for it in valid:
            fh.write(json.dumps(it, ensure_ascii=False) + "\n")
            fh.flush()
        STATS["got"] += len(valid)
        print(f"  [{level[:30]}] +{len(valid)} (total: {STATS['got']})")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=500)
    ap.add_argument("--concurrency", type=int, default=5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fh = Path(args.out).open("w", encoding="utf-8")
    n_calls = (args.target // 4) + 5
    work = [(LEVELS[i % len(LEVELS)],) for i in range(n_calls)]
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(worker, w[0], fh) for w in work]
        for f in as_completed(futs):
            f.result()
            if STATS["got"] >= args.target: break
    fh.close()
    print(f"\n[done] {STATS['got']} records in {time.time()-t0:.0f}s -> {args.out}")

if __name__ == "__main__":
    main()
