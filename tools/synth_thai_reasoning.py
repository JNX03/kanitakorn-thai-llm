"""Generate Thai-language reasoning traces for ThaiExam-style questions.

Critical addition: explicitly forces Thai reasoning, not English/Chinese.
This addresses the observed failure mode where the model thinks in English
even when the question is in Thai.

Each record:
- Thai question + 5 choices
- LONG Thai-language step-by-step reasoning (ภาษาไทยล้วน)
- Thai-language final answer "คำตอบคือ (X)"
"""
import argparse, json, subprocess, threading, time, shutil, os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = Path(__file__).resolve().parents[1]
CODEX = (shutil.which("codex") or shutil.which("codex.cmd")
         or os.path.expanduser("~/AppData/Roaming/npm/codex.cmd"))

TOPICS = [
    "วรรณคดีไทย — สุนทรภู่ พระอภัยมณี รามเกียรติ์ ขุนช้างขุนแผน อิเหนา ลิลิตพระลอ",
    "ฉันทลักษณ์ไทย — กลอนสุภาพ ฉันท์ โคลง กาพย์ยานี ๑๑ สูตรเอกโท การส่งสัมผัส",
    "ไวยากรณ์ไทย — คำลักษณนาม คำราชาศัพท์ คำซ้อน คำประสม การใช้บุพบท",
    "ประวัติศาสตร์ไทย — สุโขทัย อยุธยา ธนบุรี รัตนโกสินทร์ รัชกาล นโยบาย",
    "พุทธศาสนาในไทย — ปฏิจจสมุปบาท อริยสัจสี่ พระไตรปิฎก เถรวาท",
    "ภาษาไทยขั้นสูง — การวิเคราะห์ความหมาย เปรียบเทียบ อุปมา อุปลักษณ์ การเขียนเชิงวิชาการ",
    "ฟิสิกส์ ม.6 ในภาษาไทย — กลศาสตร์ ไฟฟ้า แสง คลื่น พลังงาน",
    "เคมี ม.6 ในภาษาไทย — อะตอม พันธะเคมี สารละลาย สมดุล กรดเบส",
    "ชีววิทยา ม.6 ในภาษาไทย — เซลล์ พันธุกรรม วิวัฒนาการ ระบบนิเวศ",
    "การลงทุน — หุ้น พันธบัตร กองทุน NPV IRR Beta Risk-return",
]

PROMPT = """สร้างข้อสอบไทยพร้อมโซลูชันแบบ chain-of-thought ภาษาไทย 5 ข้อในหัวข้อ: {topic}

**สิ่งที่สำคัญที่สุด:** การให้เหตุผลและการแก้ปัญหาต้องเป็นภาษาไทยล้วน 100% ห้ามใช้ภาษาอังกฤษหรือภาษาอื่นในการคิด

แต่ละข้อต้องมี:
1. คำถามภาษาไทยที่ชัดเจน ระดับยาก
2. ตัวเลือก 5 ตัวเลือก a, b, c, d, e เป็นภาษาไทย
3. **chain_of_thought** — โซลูชันที่ใช้เหตุผลภาษาไทยล้วน ๆ อย่างน้อย 200 คำ:
   - ขั้นที่ 1: อ่านโจทย์และทำความเข้าใจ (ภาษาไทย)
   - ขั้นที่ 2: วิเคราะห์ตัวเลือก (ภาษาไทย)
   - ขั้นที่ 3: ตัดตัวเลือกที่ผิดพร้อมเหตุผล (ภาษาไทย)
   - ขั้นที่ 4: สรุปคำตอบ (ภาษาไทย)
4. คำตอบที่ถูกต้อง (1-5)

ส่งคืน JSON:
{{
  "items": [
    {{"question": "...", "choices": [...], "chain_of_thought": "...", "answer": "3"}},
    ...
  ]
}}"""

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
    schema_path = ROOT / "synth_th_reason_schema.json"
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
        # Must have Thai chars throughout
        thai_in_q = sum(1 for c in q if '฀' <= c <= '๿') / max(len(q), 1)
        thai_in_cot = sum(1 for c in cot if '฀' <= c <= '๿') / max(len(cot), 1)
        if thai_in_q < 0.5 or thai_in_cot < 0.5: continue
        letter = chr(ord('a') + int(ans) - 1)
        # Format as SFT
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
    ap.add_argument("--target", type=int, default=500)
    ap.add_argument("--concurrency", type=int, default=5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fh = Path(args.out).open("w", encoding="utf-8")
    n_calls = (args.target // 4) + 5
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
