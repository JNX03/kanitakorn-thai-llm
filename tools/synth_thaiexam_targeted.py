"""Targeted Thai exam synth — generate MCQs in categories where R1-Distill/Qwen3-14B failed.

Categories identified from error analysis:
1. Thai poetry/prosody (กลอน, ฉันท์, โคลง, กาพย์) — rhythm/rhyme rules
2. Thai grammar / linguistics (verbs, particles, classifiers, ราชาศัพท์)
3. Thai literature (สุนทรภู่, รามเกียรติ์, classical works)
4. Thai history (Sukhothai, Ayutthaya, Rattanakosin)
5. O-NET / A-Level academic-style questions

Run:
    python3 tools/synth_thaiexam_targeted.py --target 400 --out dataset/train/train_thaiexam_targeted.jsonl
"""
import argparse, json, subprocess, threading, time, shutil, os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = Path(__file__).resolve().parents[1]
CODEX = shutil.which("codex") or shutil.which("codex.cmd") or \
        os.path.expanduser("~/AppData/Roaming/npm/codex.cmd")

CATEGORIES = {
    "thai_prosody": "บทประพันธ์ไทย (กลอนสุภาพ ฉันท์ โคลง กาพย์ยานี ๑๑) — สูตรเอกโท จำนวนคำ การส่งสัมผัส",
    "thai_grammar": "ไวยากรณ์ไทย — ชนิดของคำ คำกริยา คำลักษณนาม คำราชาศัพท์ การใช้คำบุพบท",
    "thai_lit": "วรรณคดีไทย — สุนทรภู่ พระอภัยมณี รามเกียรติ์ ขุนช้างขุนแผน ลิลิตพระลอ",
    "thai_history": "ประวัติศาสตร์ไทย — สุโขทัย อยุธยา ธนบุรี รัตนโกสินทร์ รัชกาล",
    "onet_alevel": "ข้อสอบ O-NET / A-Level ระดับมัธยมศึกษาตอนปลาย",
}

PROMPT = """สร้างข้อสอบไทยแบบปรนัย 10 ข้อ ในหัวข้อ: {category_desc}

แต่ละข้อต้องมี:
- คำถามภาษาไทยที่ชัดเจน ระดับยาก (เหมือนข้อสอบ ONET/A-Level)
- ตัวเลือก 5 ตัวเลือก a, b, c, d, e
- เฉลยที่ถูกต้องและคำอธิบายเหตุผล

ห้ามถามคำถาม meta (เกี่ยวกับไฟล์/ระบบ/AI) หรือคำถามที่ตอบไม่ได้ในเชิงข้อเท็จจริง

ส่งคืนเป็น JSON ตามรูปแบบ:
{{
  "items": [
    {{"question": "...", "choices": ["...", "...", "...", "...", "..."], "answer": "3", "explanation": "..."}},
    ...
  ]
}}"""

SCHEMA = json.dumps({
    "type":"object","properties":{"items":{"type":"array","items":{
        "type":"object","properties":{
            "question":{"type":"string"},"choices":{"type":"array","items":{"type":"string"},"minItems":5,"maxItems":5},
            "answer":{"type":"string","enum":["1","2","3","4","5"]},"explanation":{"type":"string"}
        },"required":["question","choices","answer","explanation"]
    }}},"required":["items"]
})

LOCK = threading.Lock()
STATS = {"got":0, "failed":0}

def call_codex(prompt: str) -> list:
    if not CODEX: return []
    schema_path = ROOT / "synth_codex_batch.json"
    if not schema_path.exists():
        schema_path.write_text(SCHEMA, encoding="utf-8")
    args = [CODEX,"exec","--skip-git-repo-check","--ephemeral","-s","read-only",
            "--output-schema",str(schema_path),"--color","never",
            "-c","model_reasoning_effort=\"low\"", prompt]
    try:
        proc = subprocess.run(args, capture_output=True, text=True, encoding='utf-8',
                              errors='replace', timeout=240)
        text = (proc.stdout or "") + (proc.stderr or "")
    except subprocess.TimeoutExpired: return []
    # Extract JSON
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

def filter_valid(items, category):
    out = []
    for it in items:
        q = (it.get("question") or "").strip()
        choices = it.get("choices") or []
        ans = str(it.get("answer","")).strip()
        expl = (it.get("explanation") or "").strip()
        if not q or len(choices) != 5 or ans not in {"1","2","3","4","5"}: continue
        if not any(c and len(str(c)) >= 2 for c in choices): continue
        if not expl: continue
        # Require Thai
        if not any('฀' <= c <= '๿' for c in q): continue
        # Reject meta
        if any(b in q.lower() for b in ["file","ocr","metadata","encoding","what does the file"]): continue
        out.append({"question": q, "choices": choices, "answer": ans,
                    "explanation": expl, "category": category})
    return out

def worker(category, cat_desc, fh):
    items = call_codex(PROMPT.format(category_desc=cat_desc))
    valid = filter_valid(items, category)
    with LOCK:
        for it in valid:
            fh.write(json.dumps(it, ensure_ascii=False) + "\n")
            fh.flush()
        STATS["got"] += len(valid)
        if len(valid) < 5: STATS["failed"] += 1
        print(f"  [{category}] +{len(valid)} (total: {STATS['got']})")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=400)
    ap.add_argument("--concurrency", type=int, default=5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fh = Path(args.out).open("w", encoding="utf-8")

    # Cycle through categories
    cats = list(CATEGORIES.items())
    n_calls = (args.target // 8) + 5  # each call returns ~10 records, expect ~8 valid
    work = [(cats[i % len(cats)][0], cats[i % len(cats)][1]) for i in range(n_calls)]

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(worker, cat, desc, fh) for cat, desc in work]
        for f in as_completed(futs):
            f.result()
            if STATS["got"] >= args.target: break

    fh.close()
    print(f"\n[done] {STATS['got']} records in {time.time()-t0:.0f}s -> {args.out}")

if __name__ == "__main__":
    main()
