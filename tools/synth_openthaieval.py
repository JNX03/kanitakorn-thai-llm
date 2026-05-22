"""Synthesize OpenThaiEval-style Thai academic exam questions.

OpenThaiEval = Thai standardized exams (O-NET, TGAT, A-Level, Investment Cert).
Same format as ThaiExam but broader coverage.
"""
import argparse, json, subprocess, threading, time, shutil, os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = Path(__file__).resolve().parents[1]
CODEX = (shutil.which("codex") or shutil.which("codex.cmd")
         or os.path.expanduser("~/AppData/Roaming/npm/codex.cmd"))

SUBJECTS = [
    "ฟิสิกส์ ม.ปลาย (กลศาสตร์ ไฟฟ้า แสง คลื่น)",
    "เคมี ม.ปลาย (อะตอม พันธะ สารละลาย ปฏิกิริยา)",
    "ชีววิทยา ม.ปลาย (เซลล์ พันธุกรรม ระบบร่างกาย ระบบนิเวศ)",
    "คณิตศาสตร์ ม.ปลาย (พีชคณิต แคลคูลัส ความน่าจะเป็น)",
    "ภาษาไทย ม.ปลาย (ไวยากรณ์ วรรณคดี การวิเคราะห์ บทกวี)",
    "สังคมศึกษา (ประวัติศาสตร์ ภูมิศาสตร์ เศรษฐศาสตร์ การเมือง)",
    "ภาษาอังกฤษ ม.ปลาย (ไวยากรณ์ การอ่าน คำศัพท์)",
    "ความรู้ทั่วไปเกี่ยวกับการลงทุน (หุ้น ตราสารหนี้ กองทุน ความเสี่ยง)",
]

PROMPT = """สร้างข้อสอบไทย 8 ข้อแบบ MCQ ในวิชา {subject}
ระดับยาก (O-NET / A-Level / TGAT / ใบอนุญาตการลงทุน)

แต่ละข้อต้องมี:
- คำถามภาษาไทยที่ชัดเจน
- ตัวเลือก 5 ตัวเลือก
- เฉลยที่ถูกต้อง
- คำอธิบายเหตุผล

ส่งคืน JSON:
{{
  "items": [
    {{"question": "...", "choices": ["...", "...", "...", "...", "..."], "answer": "3", "explanation": "..."}},
    ...
  ]
}}"""

SCHEMA = json.dumps({
    "type":"object","additionalProperties":False,
    "properties":{"items":{"type":"array","items":{
        "type":"object","additionalProperties":False,
        "properties":{
            "question":{"type":"string"},"choices":{"type":"array","items":{"type":"string"},"minItems":5,"maxItems":5},
            "answer":{"type":"string","enum":["1","2","3","4","5"]},"explanation":{"type":"string"}
        },"required":["question","choices","answer","explanation"]
    }}},"required":["items"]
})

LOCK = threading.Lock()
STATS = {"got":0}

def call_codex(prompt: str) -> list:
    if not CODEX: return []
    schema_path = ROOT / "synth_ote_schema.json"
    schema_path.write_text(SCHEMA, encoding="utf-8")
    args = [CODEX,"exec","--skip-git-repo-check","--ephemeral","-s","read-only",
            "--output-schema",str(schema_path),"--color","never",
            "-c","model_reasoning_effort=\"low\"", prompt]
    try:
        proc = subprocess.run(args, capture_output=True, text=True, encoding='utf-8',
                              errors='replace', timeout=240)
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

def worker(subject, fh):
    items = call_codex(PROMPT.format(subject=subject))
    valid = []
    for it in items:
        q = (it.get("question") or "").strip()
        choices = it.get("choices") or []
        ans = str(it.get("answer","")).strip()
        expl = (it.get("explanation") or "").strip()
        if not q or len(choices) != 5 or ans not in {"1","2","3","4","5"}: continue
        if not expl: continue
        if not any('฀' <= c <= '๿' for c in q): continue
        valid.append({"question": q, "choices": choices, "answer": ans,
                      "explanation": expl, "subject": subject})
    with LOCK:
        for it in valid:
            fh.write(json.dumps(it, ensure_ascii=False) + "\n")
            fh.flush()
        STATS["got"] += len(valid)
        print(f"  [{subject[:20]}] +{len(valid)} (total: {STATS['got']})")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=600)
    ap.add_argument("--concurrency", type=int, default=5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fh = Path(args.out).open("w", encoding="utf-8")
    n_calls = (args.target // 6) + 5
    work = [(SUBJECTS[i % len(SUBJECTS)],) for i in range(n_calls)]
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(worker, s[0], fh) for s in work]
        for f in as_completed(futs):
            f.result()
            if STATS["got"] >= args.target: break
    fh.close()
    print(f"\n[done] {STATS['got']} records in {time.time()-t0:.0f}s -> {args.out}")

if __name__ == "__main__":
    main()
