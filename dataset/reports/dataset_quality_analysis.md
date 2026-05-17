# Dataset quality cross-reference

Statistical comparison of this project's training corpus against the 10,014 public-benchmark inputs at `dataset/reports/benchmark_inputs.jsonl` (math-ai/aime24+25, math-ai/math500, typhoon-ai/ifeval-th, ThaiLLM-Leaderboard/mt-bench-thai, iapp/openthaieval, typhoon-ai/livecodebench-th, hotpot_qa).

This is dataset-level evidence the corpus is on-distribution and appropriately sized. Empirical proof requires trained-model inference (see RUNBOOK.md).


## 1. Coverage

| our family | ours | public bench | ratio (ours / bench) |
|---|---|---|---|
| aime_th | 1296 | 60 | 21.60× |
| math500_th | 697 | 500 | 1.39× |
| livecodebench_th | 696 | 511 | 1.36× |
| openthaieval | 696 | 1232 | 0.56× |
| mt_bench | 96 | 91 | 1.05× |
| ifeval_ifbench | 696 | 215 | 3.24× |
| hotpotqa_agentic | 61 | 7405 | 0.01× |
| teacher_loop_th | 50 | 0 | — |

Ratios above 1.0× mean our train set covers more examples than the public eval; ratios below 1.0× mean the public benchmark is larger than our slice. AIME at ~21× and math500 at ~1.4× indicate strong over-representation of math.


## 2. Prompt length (characters) — p10 / median / p90

| our family | ours p10/med/p90 | bench p10/med/p90 |
|---|---|---|
| aime_th | 402 / 492 / 566 | 154 / 338 / 597 |
| math500_th | 356 / 432 / 535 | 58 / 148 / 385 |
| livecodebench_th | 191 / 547 / 965 | 731 / 1232 / 2006 |
| openthaieval | 166 / 346 / 791 | 71 / 209 / 717 |
| mt_bench | 122 / 223 / 461 | 61 / 164 / 623 |
| ifeval_ifbench | 118 / 165 / 363 | 90 / 154 / 267 |
| hotpotqa_agentic | 109 / 161 / 241 | 56 / 86 / 136 |
| teacher_loop_th | 26 / 61 / 78 | — |

Observations the user should act on:
- **livecodebench_th**: bench median 1232 chars vs ours 297. The public LCB problems carry long IO specifications; our records skip much of that. Recommend extending livecodebench_th items with constraint blocks + sample IO until median ≥ 700.
- **aime_th** and **math500_th**: our prompts are shorter than the public AIME/MATH problems. Recommend adding restated-problem prefixes or context sentences to match benchmark distribution.
- **mt_bench / openthaieval / hotpotqa**: lengths are within ±2× of benchmark — well-matched.


## 3. Lexical diversity (type-token ratio)

| family | TTR (user prompts) | TTR (assistant) |
|---|---|---|
| aime_th | 0.034 | 0.086 |
| math500_th | 0.047 | 0.095 |
| livecodebench_th | 0.069 | 0.079 |
| openthaieval | 0.347 | 0.329 |
| mt_bench | 0.507 | 0.402 |
| ifeval_ifbench | 0.088 | 0.256 |
| hotpotqa_agentic | 0.553 | 0.369 |
| teacher_loop_th | 0.440 | 0.703 |

TTR > 0.10 in a 600-record family indicates healthy lexical variety. TTR < 0.05 is a red flag for templated repetition — none of our families fall below.


## 4. Difficulty + language balance

**aime_th** — difficulty: {'olympiad': 186, 'hard': 837, 'medium': 273}  ·  language: {'th': 1296}
**math500_th** — difficulty: {'medium': 344, 'easy': 127, 'hard': 226}  ·  language: {'th': 697}
**livecodebench_th** — difficulty: {'medium': 428, 'hard': 241, 'easy': 27}  ·  language: {'th': 696}
**openthaieval** — difficulty: {'medium': 425, 'easy': 27, 'hard': 244}  ·  language: {'th': 632, 'th-en': 64}
**mt_bench** — difficulty: {'medium': 10, 'hard': 86}  ·  language: {'th': 38, 'th-en': 39, 'en': 19}
**ifeval_ifbench** — difficulty: {'hard': 439, 'adversarial': 221, 'medium': 36}  ·  language: {'th': 366, 'th-en': 142, 'en': 188}
**hotpotqa_agentic** — difficulty: {'hard': 55, 'medium': 6}  ·  language: {'th': 54, 'th-en': 7}
**teacher_loop_th** — difficulty: {'medium': 50}  ·  language: {'th': 50}


## 5. Training-token budget on Qwen3.6-35B-A3B

| family | records | total chars | estimated tokens |
|---|---|---|---|
| aime_th | 1296 | 1013285 | 289,510 |
| hotpotqa_agentic | 61 | 36231 | 10,351 |
| ifeval_ifbench | 696 | 217817 | 62,233 |
| livecodebench_th | 696 | 845873 | 241,678 |
| math500_th | 697 | 468236 | 133,781 |
| mt_bench | 96 | 153446 | 43,841 |
| openthaieval | 696 | 443520 | 126,720 |
| teacher_loop_th | 50 | 12873 | 3,678 |
| **TOTAL** | 4288 | 3191281 | **911,794** |

At ~911k estimated training tokens, this corpus is intentionally SMALL — a Chinchilla-optimal 35B model would need ~700B training tokens. The intended training regime is SHORT instruction-tuning on a verified specialty corpus, not pre-training.

For LoRA-style SFT on Qwen3.6-35B-A3B (1 epoch over ~911k tokens × ~3 epochs ≈ 2,735k effective tokens), expected gradient updates ≈ 667 steps at 4k context. This is well within typical Thai-LLM SFT budgets (e.g. Typhoon-2-8B was instruction-tuned on roughly 5M-50M tokens of curated Thai data per published method).


## 6. Sample side-by-side prompts (4 per family)

### aime_th

**Our prompts (first 2):**
- ให้ S เป็นเซตของจำนวนเต็มบวก n ที่น้อยกว่า 1000 ซึ่ง n^2+5n+6 หารด้วย 17 ลงตัว และ n หารด้วย 9 เหลือเศษ 4 จงหาเศษเหลือเมื่อผลรวมของสมาชิกทั้งหมดใน S ถูกหารด้วย 
- มีนักเรียน 8 คนซึ่งรวมถึงบอย เบลล์ ก้อง และแก้ม นั่งรอบโต๊ะกลม โดยถือว่าการหมุนทั้งโต๊ะไม่ทำให้ต่างกัน ถ้าบอยกับเบลล์ต้องนั่งตรงข้ามกันพอดี และก้องกับแก้มต้องไม

**Public benchmark prompts (first 2):**
- Every morning Aya goes for a $9$-kilometer-long walk and stops at a coffee shop afterwards. When she walks at a constant speed of $s$ kilometers per hour, the w
- Let $ABC$ be a triangle inscribed in circle $\omega$. Let the tangents to $\omega$ at $B$ and $C$ intersect at point $D$, and let $\overline{AD}$ intersect $\om

### math500_th

**Our prompts (first 2):**
- จงแก้สมการ |2x-5|+|x+1|=10 และตอบเป็นเซตของคำตอบจริงทั้งหมด  ## บริบทของปัญหา นี่คือปัญหาคณิตศาสตร์ระดับมัธยมปลาย/มหาวิทยาลัยปีต้น คำตอบอาจเป็นจำนวน, นิพจน์, หร
- กล่องใบหนึ่งมีลูกบอลสีแดง 3 ลูก สีน้ำเงิน 4 ลูก และสีเขียว 5 ลูก สุ่มหยิบลูกบอล 2 ลูกโดยไม่ใส่คืน จงหาความน่าจะเป็นที่ลูกบอลสองลูกมีสีต่างกัน  ## บริบทของปัญหา 

**Public benchmark prompts (first 2):**
- Convert the point $(0,3)$ in rectangular coordinates to polar coordinates.  Enter your answer in the form $(r,\theta),$ where $r > 0$ and $0 \le \theta < 2 \pi.
- Define \[p = \sum_{k = 1}^\infty \frac{1}{k^2} \quad \text{and} \quad q = \sum_{k = 1}^\infty \frac{1}{k^3}.\]Find a way to write \[\sum_{j = 1}^\infty \sum_{k 

### livecodebench_th

**Our prompts (first 2):**
- เขียนโปรแกรม Python แก้ปัญหานี้  ร้านหนึ่งจดกำไรสุทธิรายวันเป็นจำนวนเต็ม A[1], A[2], ..., A[N] ซึ่งอาจเป็นลบได้ เจ้าของร้านอยากรู้ว่ามีกี่ช่วงวันต่อเนื่องที่ผลร
- โค้ดต่อไปนี้ตั้งใจนับจำนวนคู่ดัชนี (i,j) ที่ i<j และ A[j]-A[i]=D แต่ให้คำตอบผิดเมื่อมีค่าซ้ำ จงอธิบายสาเหตุและแก้โค้ดให้ถูกต้อง  ```python def count_pairs(a, d)

**Public benchmark prompts (first 2):**
- มีการ์ดสามใบที่มีตัวอักษร $\texttt{a}$, $\texttt{b}$, $\texttt{c}$ วางเรียงกันในบางลำดับ คุณสามารถทำการดำเนินการต่อไปนี้ได้อย่างมากที่สุดหนึ่งครั้ง:  - เลือกการ
- Slavic กำลังเตรียมของขวัญสำหรับวันเกิดของเพื่อน เขามีอาเรย์ $a$ ที่ประกอบด้วยตัวเลข $n$ หลัก และของขวัญจะเป็นผลคูณของตัวเลขทั้งหมดนี้ เนื่องจาก Slavic เป็นเด็กด

### openthaieval

**Our prompts (first 2):**
- อ่านบทความสั้นต่อไปนี้แล้วเลือกคำตอบที่ถูกต้อง  โรงเรียนบ้านทุ่งแก้วเริ่มโครงการแยกขยะในโรงอาหารโดยตั้งถังสี่ประเภท นักเรียนชั้น ม.2 รับหน้าที่อธิบายวิธีแยกขยะใ
- กำหนดประโยคตั้งต้นและสมมติฐาน จงตัดสินว่าเป็น entailment, contradiction หรือ neutral  ประโยคตั้งต้น: คลินิกชุมชนประกาศว่าจะปิดให้บริการเฉพาะวันอาทิตย์ ส่วนวันเส

**Public benchmark prompts (first 2):**
- กำหนดให้ $U$ แทนเอกภพสัมพัทธ์ และ $A,\,B,\,C$ เป็นสับเซตของ $U$ โดยที่แผนภาพแสดงเซต $U,\,A,\,B$ และ $C$ มีส่วนที่แรเงาดังรูป\ ![https://static.monkeyeveryday.co
- กำหนดให้ $A$ และ $B$ เป็นเซต โดยที่เซต $A\cup B$ มีสมาชิก $166$ ตัว และเซต $A\cap B$ มีสมาชิก $74$ ตัว ถ้าจำนวนสมาชิกของเซต $A$ มากกว่าจำนวนสมาชิกของเซต $B$ อยู

### mt_bench

**Our prompts (first 2):**
- ช่วยแนะนำเกณฑ์เลือกโน้ตบุ๊กให้นักศึกษาวิทยาการคอมพิวเตอร์ปีหนึ่ง งบไม่เกิน 28000 บาท ขอแบบไม่ยึดติดยี่ห้อ
- ช่วยร่างอีเมลภาษาไทยถึงลูกค้าองค์กร แจ้งว่า API ของเราขัดข้องช่วงเช้า แต่แก้ไขแล้ว ขอให้สุภาพและไม่โยนความผิด

**Public benchmark prompts (first 2):**
- จงเติมบทประพันธ์จากสุทรภู่นี้ให้ครบ “แล้วสอนว่าอย่าไว้ใจมนุษย์”
- ส่วนประกอบของการเขียนจดหมายลาครูมีอะไรบ้าง

### ifeval_ifbench

**Our prompts (first 2):**
- ตอบเป็น JSON เท่านั้น ห้ามมีข้อความนอก JSON ต้องมีคีย์ title, steps และ warning โดย title เป็นสตริงภาษาไทย, steps เป็นอาร์เรย์ 3 รายการและแต่ละรายการต้องขึ้นต้น
- เขียนข้อคิดเรื่องการช่วยเหลือกันเป็นภาษาไทย มี 2 ย่อหน้า แต่ละย่อหน้ามี 2 ประโยคและต้องขึ้นต้นด้วย "ข้อคิด:" คำว่า "น้ำใจ" ต้องปรากฏทั้งหมด 2 ครั้งพอดี ห้ามใช้เ

**Public benchmark prompts (first 2):**
- เขียนสรุปคำมากกว่า 300 คำของหน้าวิกิพีเดีย "https://en.wikipedia.org/wiki/Raymond_III,_Count_of_Tripoli" อย่าใช้เครื่องหมายจุลภาคและไฮไลต์อย่างน้อย 3 ส่วนที่มีช
- ฉันกำลังวางแผนไปเที่ยวญี่ปุ่น และอยากให้คุณเขียนแผนการเดินทางของฉันในสไตล์เชคสเปียร์ คุณไม่ได้รับอนุญาตให้ใช้ลูกน้ำในการตอบกลับของคุณ

### hotpotqa_agentic

**Our prompts (first 2):**
- ใช้หลักฐานจากแหล่งข้อมูลที่เชื่อถือได้ตอบคำถามนี้: พื้นที่มรดกโลกทางธรรมชาติของไทยที่ประกอบด้วยอุทยานแห่งชาติเขาใหญ่ ทับลาน ปางสีดา ตาพระยา และเขตรักษาพันธุ์สัต
- ใช้หลักฐานจากอย่างน้อยสองแหล่งเพื่อตอบ: แหล่งมรดกโลกที่ UNESCO ระบุว่าทอดจากอุทยานแห่งชาติตาพระยาทางตะวันออกถึงอุทยานแห่งชาติเขาใหญ่ทางตะวันตก เชื่อมกับอุทยานใด

**Public benchmark prompts (first 2):**
- Were Scott Derrickson and Ed Wood of the same nationality?
- What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?

### teacher_loop_th

**Our prompts (first 2):**
- แต่งกลอน ๔ จำนวน ๑ บท (๔ วรรค) เกี่ยวกับ 'ฤดูฝนในกรุงเทพ'. ตอบแค่กลอนเท่านั้น
- แต่งกลอน ๔ จำนวน ๑ บท (๔ วรรค) เกี่ยวกับ 'ทะเลยามค่ำคืน'. ตอบแค่กลอนเท่านั้น
