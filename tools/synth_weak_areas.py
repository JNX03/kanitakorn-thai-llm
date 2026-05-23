"""Targeted synth for ThaiExam weak areas: investment, history, grammar."""
import argparse, json, subprocess, threading, time, shutil, os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = Path(__file__).resolve().parents[1]
CODEX = (shutil.which("codex") or shutil.which("codex.cmd")
         or os.path.expanduser("~/AppData/Roaming/npm/codex.cmd"))

TOPICS = [
    "การลงทุนและการเงิน — หุ้น พันธบัตร กองทุน ความเสี่ยง NPV IRR Beta ตราสารอนุพันธ์ ภาษีการลงทุน",
    "ใบอนุญาตการลงทุน ก-สรว — ความรู้พื้นฐาน ใบ Single License IC P-1 P-2 P-3 ETF MAI",
    "ประวัติศาสตร์ไทย สมัยสุโขทัย — พ่อขุนรามคำแหง ลายสือไทย ศิลาจารึก เศรษฐกิจ การปกครอง",
    "ประวัติศาสตร์ไทย สมัยอยุธยา — รัชกาล สงครามพม่า การค้าต่างประเทศ เศรษฐกิจ ศิลปะ",
    "ประวัติศาสตร์ไทย รัตนโกสินทร์ — รัชกาลที่ 1-10 ตำแหน่ง นโยบาย เหตุการณ์สำคัญ",
    "ไวยากรณ์ไทยขั้นสูง — คำลักษณนาม ราชาศัพท์ คำซ้อน คำประสม สำนวน สุภาษิต",
    "ภูมิศาสตร์ไทย — ภาคต่างๆ จังหวัด สภาพภูมิอากาศ ทรัพยากร เศรษฐกิจภูมิภาค",
    "พุทธศาสนา/ศาสนา — อริยสัจ 4 ปฏิจจสมุปบาท พุทธประวัติ พระไตรปิฎก เถรวาท",
]

PROMPT = """สร้างข้อสอบไทย MCQ 6 ข้อ ในหัวข้อ: {topic}
ระดับยาก (O-NET/A-Level/Single License/ใบอนุญาตการลงทุน)

แต่ละข้อต้องมี:
- คำถามภาษาไทยที่ชัดเจน รายละเอียดเฉพาะ
- ตัวเลือก 5 ตัวเลือก
- chain_of_thought (เหตุผลภาษาไทยล้วน อย่างน้อย 150 คำ)
- คำตอบที่ถูกต้อง

ส่งคืน JSON: {{ "items": [{{"question":"...","choices":[...],"chain_of_thought":"...","answer":"3"}}] }}"""

SCHEMA = json.dumps({
    "type":"object","additionalProperties":False,
    "properties":{"items":{"type":"array","items":{
        "type":"object","additionalProperties":False,
        "properties":{
            "question":{"type":"string"},"choices":{"type":"array","items":{"type":"string"},"minItems":5,"maxItems":5},
            "chain_of_thought":{"type":"string"},
            "answer":{"type":"string","enum":["1","2","3","4","5"]}
        },"required":["question","choices","chain_of_thought","answer"]
    }}},"required":["items"]
})

LOCK = threading.Lock()
STATS = {"got":0}

def call_codex(prompt: str) -> list:
    if not CODEX: return []
    schema_path = ROOT / "synth_weak_schema.json"
    schema_path.write_text(SCHEMA, encoding="utf-8")
    args = [CODEX,"exec","--skip-git-repo-check","--ephemeral","-s","read-only",
            "--output-schema",str(schema_path),"--color","never",
            "-c","model_reasoning_effort=\"medium\"", prompt]
    try:
        proc = subprocess.run(args, capture_output=True, text=True, encoding='utf-8',
                              errors='replace', timeout=400)
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

def worker(topic, fh):
    items = call_codex(PROMPT.format(topic=topic))
    valid = []
    for it in items:
        q = (it.get("question") or "").strip()
        choices = it.get("choices") or []
        cot = (it.get("chain_of_thought") or "").strip()
        ans = str(it.get("answer","")).strip()
        if not q or len(choices) != 5 or not cot or ans not in {"1","2","3","4","5"}: continue
        thai_in_cot = sum(1 for c in cot if '฀' <= c <= '๿') / max(len(cot), 1)
        if thai_in_cot < 0.5: continue
        letter = chr(ord('a') + int(ans) - 1)
        user = q + "\n\nตัวเลือก:\n" + "\n".join(f"({chr(97+i)}) {c}" for i, c in enumerate(choices))
        user += "\n\nคิดและให้เหตุผลเป็นภาษาไทย แล้วระบุคำตอบในรูปแบบ คำตอบคือ (X)"
        assistant = cot + f"\n\nคำตอบคือ ({letter})"
        valid.append({"messages":[
            {"role":"user","content":user},
            {"role":"assistant","content":assistant},
        ], "topic": topic[:40]})
    with LOCK:
        for it in valid:
            fh.write(json.dumps(it, ensure_ascii=False) + "\n")
            fh.flush()
        STATS["got"] += len(valid)
        print(f"  [{topic[:30]}] +{len(valid)} (total: {STATS['got']})")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=600)
    ap.add_argument("--concurrency", type=int, default=5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fh = Path(args.out).open("w", encoding="utf-8")
    n_calls = (args.target // 5) + 5
    work = [(TOPICS[i % len(TOPICS)],) for i in range(n_calls)]
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
