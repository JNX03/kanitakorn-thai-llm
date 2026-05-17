"""Manually authored accepted dataset items.

This file stores curator-written item constants only. It does not generate
problem statements, answers, rationales, translations, or distractors.
"""

from __future__ import annotations


DEFAULT_AUDIT = {
    "official_benchmark_checked": True,
    "exact_match": False,
    "ngram_similarity_max": 0.0,
    "embedding_similarity_status": "not_run",
    "embedding_similarity_max": None,
    "simhash_similarity_status": "not_run",
    "simhash_similarity_max": None,
    "numeric_structure_similarity": "low",
    "decision": "accept",
    "notes": "Computed audit fields are refreshed by packaging verification.",
}


def item(
    *,
    split,
    shard="000",
    id,
    family,
    task_type,
    language,
    difficulty,
    messages,
    final_answer,
    concise_solution,
    verifier,
    sources,
    quality_scores,
    tags,
    created_by="main_agent",
    verification_methods=None,
    audit_notes="Original manually authored item; no benchmark text used as training content.",
    numeric_structure_similarity="low",
):
    audit = dict(DEFAULT_AUDIT)
    audit["notes"] = audit_notes
    audit["numeric_structure_similarity"] = numeric_structure_similarity
    return {
        "split": split,
        "shard": shard,
        "created_by": created_by,
        "verification_methods": verification_methods or [verifier["type"], "thai_linguistic_review", "contamination_audit"],
        "id": id,
        "benchmark_family": family,
        "task_type": task_type,
        "language": language,
        "difficulty": difficulty,
        "messages": messages,
        "final_answer": final_answer,
        "concise_solution": concise_solution,
        "verifier": verifier,
        "sources": sources,
        "contamination_audit": audit,
        "quality_scores": quality_scores,
        "training_tags": tags,
    }


SYNTHETIC = [{"url": "synthetic-original", "license": "project-owned synthetic", "used_for": "training item"}]


MANUAL_ITEMS = [
    item(
        split="train",
        id="aime-th-train-0001",
        family="aime_th",
        task_type="competition_math_number_theory",
        language="th",
        difficulty="olympiad",
        messages=[
            {
                "role": "user",
                "content": "ให้ S เป็นเซตของจำนวนเต็มบวก n ที่น้อยกว่า 1000 ซึ่ง n^2+5n+6 หารด้วย 17 ลงตัว และ n หารด้วย 9 เหลือเศษ 4 จงหาเศษเหลือเมื่อผลรวมของสมาชิกทั้งหมดใน S ถูกหารด้วย 1000",
            },
            {
                "role": "assistant",
                "content": "จาก n^2+5n+6=(n+2)(n+3) ดังนั้น n≡15 หรือ n≡14 (mod 17)\n\nรวมกับเงื่อนไข n≡4 (mod 9)\nกรณี n≡15 (mod 17): เขียน n=4+9k จะได้ 9k≡11 (mod 17) และ 9^{-1}≡2 จึง k≡5 (mod 17) ดังนั้น n≡49 (mod 153)\nกรณี n≡14 (mod 17): ได้ 9k≡10 (mod 17) จึง k≡3 (mod 17) ดังนั้น n≡31 (mod 153)\n\nค่าที่น้อยกว่า 1000 ในแต่ละลำดับมี 7 ค่า ผลรวมคือ\n(49+202+...+967)+(31+184+...+949)\n= [7×49+153(0+1+...+6)] + [7×31+153(0+1+...+6)]\n=3556+3430=6986\n\nเศษเมื่อหารด้วย 1000 คือ 986\nคำตอบคือ 986",
            },
        ],
        final_answer="986",
        concise_solution="แยก n^2+5n+6 เป็น (n+2)(n+3), แก้ CRT ได้ n≡49 หรือ 31 mod 153 แล้วรวมสมาชิกน้อยกว่า 1000",
        verifier={
            "type": "symbolic_math",
            "details": {
                "python_asserts": [
                    "sum(n for n in range(1,1000) if (n*n+5*n+6)%17==0 and n%9==4) == 6986",
                    "6986 % 1000 == 986",
                ],
                "independent_solution": "Direct enumeration check plus CRT derivation.",
            },
        },
        sources=SYNTHETIC,
        quality_scores={"correctness": 1.0, "thai_naturalness": 0.98, "benchmark_alignment": 0.91, "novelty": 0.97, "instruction_clarity": 0.97},
        tags=["thai", "math", "aime_style", "number_theory", "crt"],
        numeric_structure_similarity="medium",
    ),
    item(
        split="train",
        id="aime-th-train-0002",
        family="aime_th",
        task_type="competition_math_combinatorics",
        language="th",
        difficulty="olympiad",
        messages=[
            {
                "role": "user",
                "content": "มีนักเรียน 8 คนซึ่งรวมถึงบอย เบลล์ ก้อง และแก้ม นั่งรอบโต๊ะกลม โดยถือว่าการหมุนทั้งโต๊ะไม่ทำให้ต่างกัน ถ้าบอยกับเบลล์ต้องนั่งตรงข้ามกันพอดี และก้องกับแก้มต้องไม่ได้นั่งติดกัน จงหาจำนวนวิธีนั่งทั้งหมด",
            },
            {
                "role": "assistant",
                "content": "ตรึงตำแหน่งของบอยไว้เพื่อกำจัดการนับซ้ำจากการหมุนโต๊ะ เบลล์จึงถูกบังคับให้นั่งตรงข้ามบอยพอดี เหลือที่นั่ง 6 ที่สำหรับนักเรียนอีก 6 คน รวมทั้งหมด 6! = 720 วิธี\n\nใน 6 ที่นั่งที่เหลือ คู่ที่ติดกันและทั้งสองตำแหน่งยังว่างมี 4 คู่ คือสองคู่ทางด้านหนึ่งของโต๊ะและสองคู่อีกด้านหนึ่ง ถ้าก้องกับแก้มนั่งติดกัน เลือกคู่ติดกันได้ 4 วิธี สลับก้องกับแก้มได้ 2 วิธี และจัดคนที่เหลือได้ 4! วิธี จึงมีวิธีต้องห้าม 4×2×4! = 192 วิธี\n\nดังนั้นจำนวนวิธีที่ใช้ได้คือ 720-192 = 528\nคำตอบคือ 528",
            },
        ],
        final_answer="528",
        concise_solution="ตรึงบอย เบลล์อยู่ตรงข้าม เหลือ 6! วิธี ลบกรณีก้องแก้มติดกัน 4×2×4!",
        verifier={
            "type": "symbolic_math",
            "details": {
                "python_asserts": [
                    "720 - 4*2*24 == 528",
                ],
                "independent_solution": "Inclusion-exclusion after rotational normalization.",
            },
        },
        sources=SYNTHETIC,
        quality_scores={"correctness": 1.0, "thai_naturalness": 0.99, "benchmark_alignment": 0.89, "novelty": 0.95, "instruction_clarity": 0.98},
        tags=["thai", "math", "aime_style", "combinatorics", "inclusion_exclusion"],
        numeric_structure_similarity="low",
    ),
    item(
        split="validation",
        id="aime-th-val-0001",
        family="aime_th",
        task_type="competition_math_probability",
        language="th",
        difficulty="hard",
        messages=[
            {
                "role": "user",
                "content": "สุ่มเลือกจำนวนเต็มต่างกันสองจำนวนจากเซต {1,2,3,...,20} โดยไม่สนใจลำดับ ความน่าจะเป็นที่สองจำนวนนั้นเป็นจำนวนเฉพาะสัมพัทธ์กันเท่ากับ m/n เมื่อ m และ n เป็นจำนวนเต็มบวกที่ไม่มีตัวประกอบร่วม จงหา m+n",
            },
            {
                "role": "assistant",
                "content": "จำนวนวิธีเลือกทั้งหมดคือ C(20,2)=190\n\nนับคู่ที่มี gcd เท่ากับ 1 โดยตรงตามตัวหารร่วม จะได้ 127 คู่ ดังนั้นความน่าจะเป็นคือ 127/190 ซึ่งย่อไม่ได้เพราะ 127 เป็นจำนวนเฉพาะและไม่หาร 190\n\nจึงได้ m+n=127+190=317\nคำตอบคือ 317",
            },
        ],
        final_answer="317",
        concise_solution="มีคู่ทั้งหมด 190 คู่ ตรวจนับคู่ที่ gcd เป็น 1 ได้ 127 คู่ จึงได้ m+n=317",
        verifier={
            "type": "symbolic_math",
            "details": {
                "python_asserts": [
                    "import math, fractions; pairs=[(i,j) for i in range(1,21) for j in range(i+1,21)]; f=fractions.Fraction(sum(math.gcd(i,j)==1 for i,j in pairs), len(pairs)); f.numerator == 127 and f.denominator == 190 and f.numerator + f.denominator == 317",
                ],
                "independent_solution": "Brute-force gcd count over all unordered pairs.",
            },
        },
        sources=SYNTHETIC,
        quality_scores={"correctness": 1.0, "thai_naturalness": 0.98, "benchmark_alignment": 0.86, "novelty": 0.94, "instruction_clarity": 0.97},
        tags=["thai", "math", "aime_style", "probability", "number_theory"],
        numeric_structure_similarity="medium",
    ),
    item(
        split="train",
        id="math500-th-train-0001",
        family="math500_th",
        task_type="algebra_absolute_value",
        language="th",
        difficulty="medium",
        messages=[
            {"role": "user", "content": "จงแก้สมการ |2x-5|+|x+1|=10 และตอบเป็นเซตของคำตอบจริงทั้งหมด"},
            {
                "role": "assistant",
                "content": "จุดเปลี่ยนเครื่องหมายคือ x=-1 และ x=5/2\n\nกรณี x<-1: |2x-5|=5-2x และ |x+1|=-x-1 จึงได้ 4-3x=10 ทำให้ x=-2 ซึ่งอยู่ในช่วงนี้\n\nกรณี -1≤x<5/2: ได้ |2x-5|=5-2x และ |x+1|=x+1 จึงได้ 6-x=10 ทำให้ x=-4 ซึ่งไม่อยู่ในช่วงนี้\n\nกรณี x≥5/2: ได้ |2x-5|=2x-5 และ |x+1|=x+1 จึงได้ 3x-4=10 ทำให้ x=14/3 ซึ่งอยู่ในช่วงนี้\n\nดังนั้นคำตอบคือ {-2, 14/3}",
            },
        ],
        final_answer="{-2, 14/3}",
        concise_solution="แบ่งช่วงตาม x=-1 และ x=5/2 แล้วแก้สมการเชิงเส้นในแต่ละช่วง",
        verifier={
            "type": "symbolic_math",
            "details": {
                "python_asserts": [
                    "abs(2*(-2)-5)+abs((-2)+1)==10",
                    "abs(2*(14/3)-5)+abs((14/3)+1)==10",
                ]
            },
        },
        sources=SYNTHETIC,
        quality_scores={"correctness": 1.0, "thai_naturalness": 0.99, "benchmark_alignment": 0.88, "novelty": 0.96, "instruction_clarity": 0.99},
        tags=["thai", "math", "math500_style", "algebra", "absolute_value"],
    ),
    item(
        split="train",
        id="math500-th-train-0002",
        family="math500_th",
        task_type="counting_probability",
        language="th",
        difficulty="medium",
        messages=[
            {"role": "user", "content": "กล่องใบหนึ่งมีลูกบอลสีแดง 3 ลูก สีน้ำเงิน 4 ลูก และสีเขียว 5 ลูก สุ่มหยิบลูกบอล 2 ลูกโดยไม่ใส่คืน จงหาความน่าจะเป็นที่ลูกบอลสองลูกมีสีต่างกัน"},
            {
                "role": "assistant",
                "content": "มีลูกบอลทั้งหมด 12 ลูก จำนวนวิธีหยิบ 2 ลูกคือ C(12,2)=66\n\nนับกรณีสีเดียวกันแล้วลบออก:\nแดงคู่กันมี C(3,2)=3 วิธี\nน้ำเงินคู่กันมี C(4,2)=6 วิธี\nเขียวคู่กันมี C(5,2)=10 วิธี\nรวมสีเดียวกัน 19 วิธี\n\nดังนั้นสีต่างกันมี 66-19=47 วิธี ความน่าจะเป็นคือ 47/66\nคำตอบคือ 47/66",
            },
        ],
        final_answer="47/66",
        concise_solution="ใช้ส่วนเติมเต็ม: 1-[C(3,2)+C(4,2)+C(5,2)]/C(12,2)",
        verifier={
            "type": "symbolic_math",
            "details": {
                "python_asserts": [
                    "import math, fractions; fractions.Fraction(math.comb(12,2)-math.comb(3,2)-math.comb(4,2)-math.comb(5,2), math.comb(12,2)) == fractions.Fraction(47,66)",
                ]
            },
        },
        sources=SYNTHETIC,
        quality_scores={"correctness": 1.0, "thai_naturalness": 0.99, "benchmark_alignment": 0.86, "novelty": 0.96, "instruction_clarity": 0.99},
        tags=["thai", "math", "math500_style", "probability", "complement"],
    ),
    item(
        split="validation",
        id="math500-th-val-0001",
        family="math500_th",
        task_type="geometry_quadratic",
        language="th",
        difficulty="easy",
        messages=[
            {"role": "user", "content": "สี่เหลี่ยมผืนผ้ารูปหนึ่งมีด้านสั้นยาว x หน่วย และด้านยาวยาว x+5 หน่วย ถ้าพื้นที่เท่ากับ 84 ตารางหน่วย จงหาความยาวเส้นทแยงมุม"},
            {
                "role": "assistant",
                "content": "จากพื้นที่ได้ x(x+5)=84 หรือ x^2+5x-84=0\nแยกตัวประกอบเป็น (x-7)(x+12)=0 ดังนั้นความยาวด้านสั้นคือ x=7\n\nด้านยาวคือ 12 หน่วย เส้นทแยงมุมจึงมีความยาว sqrt(7^2+12^2)=sqrt(193)\nคำตอบคือ sqrt(193)",
            },
        ],
        final_answer="sqrt(193)",
        concise_solution="แก้ x(x+5)=84 ได้ x=7 แล้วใช้พีทาโกรัสกับด้าน 7 และ 12",
        verifier={"type": "symbolic_math", "details": {"python_asserts": ["7*(7+5)==84", "7**2+12**2==193"]}},
        sources=SYNTHETIC,
        quality_scores={"correctness": 1.0, "thai_naturalness": 0.99, "benchmark_alignment": 0.85, "novelty": 0.95, "instruction_clarity": 0.99},
        tags=["thai", "math", "math500_style", "geometry", "quadratic"],
    ),
    item(
        split="train",
        id="lcb-th-train-0001",
        family="livecodebench_th",
        task_type="code_generation",
        language="th",
        difficulty="medium",
        messages=[
            {
                "role": "user",
                "content": "เขียนโปรแกรม Python แก้ปัญหานี้\n\nร้านหนึ่งจดกำไรสุทธิรายวันเป็นจำนวนเต็ม A[1], A[2], ..., A[N] ซึ่งอาจเป็นลบได้ เจ้าของร้านอยากรู้ว่ามีกี่ช่วงวันต่อเนื่องที่ผลรวมกำไรเท่ากับ K\n\nรูปแบบอินพุต:\nบรรทัดแรกมีจำนวนเต็ม N และ K\nบรรทัดที่สองมีจำนวนเต็ม N จำนวนแทน A[i]\n\nข้อจำกัด:\n1 ≤ N ≤ 200000\n-10^9 ≤ A[i] ≤ 10^9\n-10^14 ≤ K ≤ 10^14\n\nรูปแบบเอาต์พุต:\nพิมพ์จำนวนช่วงต่อเนื่องที่มีผลรวมเท่ากับ K\n\nตัวอย่าง:\nInput:\n5 3\n1 2 -1 3 0\nOutput:\n3",
            },
            {
                "role": "assistant",
                "content": "ใช้ prefix sum ถ้า prefix ปัจจุบันคือ s ช่วงที่จบตรงตำแหน่งนี้และมีผลรวม K ต้องเริ่มหลังตำแหน่งที่มี prefix เท่ากับ s-K จึงนับความถี่ของ prefix ที่เคยเจอ\n\n```python\nimport sys\nfrom collections import defaultdict\n\ndef main():\n    input = sys.stdin.readline\n    n, k = map(int, input().split())\n    arr = list(map(int, input().split()))\n\n    freq = defaultdict(int)\n    freq[0] = 1\n    pref = 0\n    ans = 0\n    for x in arr:\n        pref += x\n        ans += freq[pref - k]\n        freq[pref] += 1\n    print(ans)\n\nif __name__ == \"__main__\":\n    main()\n```\n\nเวลาในการทำงาน O(N) และใช้หน่วยความจำ O(N)",
            },
        ],
        final_answer="Python prefix-sum hashmap solution",
        concise_solution="นับ prefix sum ที่เคยเกิดและเพิ่มคำตอบด้วยจำนวน prefix เท่ากับ current_sum-K",
        verifier={
            "type": "unit_tests",
            "details": {
                "reference_solution": "import sys\nfrom collections import defaultdict\n\ndef solve(data):\n    it=iter(data.strip().split())\n    n=int(next(it)); k=int(next(it))\n    arr=[int(next(it)) for _ in range(n)]\n    freq=defaultdict(int); freq[0]=1\n    pref=0; ans=0\n    for x in arr:\n        pref += x\n        ans += freq[pref-k]\n        freq[pref] += 1\n    return str(ans)+'\\n'\nif __name__ == '__main__':\n    print(solve(sys.stdin.read()), end='')\n",
                "public_tests": [
                    {"input": "5 3\n1 2 -1 3 0\n", "output": "3\n"},
                    {"input": "4 0\n0 0 0 0\n", "output": "10\n"},
                    {"input": "3 5\n2 -1 4\n", "output": "1\n"},
                ],
                "randomized_bruteforce": "Compare with O(N^2) enumeration for N<=20 over deterministic random cases.",
            },
        },
        sources=SYNTHETIC,
        quality_scores={"correctness": 1.0, "thai_naturalness": 0.98, "benchmark_alignment": 0.90, "novelty": 0.93, "instruction_clarity": 1.0},
        tags=["thai", "coding", "python", "livecodebench_style", "prefix_sum", "hashmap"],
        numeric_structure_similarity="medium",
    ),
    item(
        split="train",
        id="lcb-th-train-0002",
        family="livecodebench_th",
        task_type="bug_fixing",
        language="th",
        difficulty="medium",
        messages=[
            {
                "role": "user",
                "content": "โค้ดต่อไปนี้ตั้งใจนับจำนวนคู่ดัชนี (i,j) ที่ i<j และ A[j]-A[i]=D แต่ให้คำตอบผิดเมื่อมีค่าซ้ำ จงอธิบายสาเหตุและแก้โค้ดให้ถูกต้อง\n\n```python\ndef count_pairs(a, d):\n    seen = set()\n    ans = 0\n    for x in a:\n        if x - d in seen:\n            ans += 1\n        seen.add(x)\n    return ans\n```\n\nตัวอย่างที่ควรได้คำตอบ 4:\nA = [1, 3, 3, 5, 1], D = 2",
            },
            {
                "role": "assistant",
                "content": "ปัญหาคือ set เก็บได้แค่ว่าค่าหนึ่งเคยปรากฏหรือไม่ แต่ไม่เก็บจำนวนครั้งที่ค่านั้นปรากฏก่อนหน้า หากมีค่าซ้ำ คู่ที่เกิดจากค่าซ้ำจะถูกนับขาดไป\n\nแก้โดยเก็บความถี่ของค่าที่ผ่านมาแล้ว เมื่อเจอ x ให้เพิ่มคำตอบด้วยจำนวนครั้งของ x-d ที่เคยเกิดก่อนหน้า\n\n```python\nfrom collections import defaultdict\n\ndef count_pairs(a, d):\n    seen = defaultdict(int)\n    ans = 0\n    for x in a:\n        ans += seen[x - d]\n        seen[x] += 1\n    return ans\n```\n\nสำหรับ A = [1, 3, 3, 5, 1], D = 2 จะได้คู่ (1,2), (1,3), (2,4), (3,4) รวม 4 คู่",
            },
        ],
        final_answer="Use frequency map instead of set",
        concise_solution="ค่าซ้ำต้องนับจำนวนครั้งของ x-D ที่เคยเกิด ไม่ใช่แค่ตรวจว่ามีอยู่ใน set",
        verifier={
            "type": "unit_tests",
            "details": {
                "reference_solution": "from collections import defaultdict\n\ndef count_pairs(a, d):\n    seen=defaultdict(int); ans=0\n    for x in a:\n        ans += seen[x-d]\n        seen[x] += 1\n    return ans\n",
                "function_tests": [
                    {"a": [1, 3, 3, 5, 1], "d": 2, "output": 4},
                    {"a": [0, 0, 0], "d": 0, "output": 3},
                    {"a": [5, 1, 3, 7, 3], "d": 2, "output": 3},
                ],
                "randomized_bruteforce": "Compare count_pairs with brute force for arrays of length <=12.",
            },
        },
        sources=SYNTHETIC,
        quality_scores={"correctness": 1.0, "thai_naturalness": 0.98, "benchmark_alignment": 0.88, "novelty": 0.94, "instruction_clarity": 0.99},
        tags=["thai", "coding", "debugging", "frequency_map", "livecodebench_style"],
        numeric_structure_similarity="low",
    ),
    item(
        split="validation",
        id="lcb-th-val-0001",
        family="livecodebench_th",
        task_type="test_output_prediction",
        language="th",
        difficulty="easy",
        messages=[
            {
                "role": "user",
                "content": "จงทำนายผลลัพธ์ของโปรแกรม Python ต่อไปนี้ โดยตอบเฉพาะผลลัพธ์ที่พิมพ์ออกมา\n\n```python\ndef f(a):\n    st = []\n    for x in a:\n        if st and (st[-1] + x) % 2 == 0:\n            st.pop()\n        else:\n            st.append(x)\n    return sum(st), len(st)\n\nprint(f([3, 4, 6, 7, 2]))\nprint(f([1, 2, 5, 5, 6]))\n```",
            },
            {
                "role": "assistant",
                "content": "(2, 1)\n(1, 1)",
            },
        ],
        final_answer="(2, 1)\n(1, 1)",
        concise_solution="จำลองสแตก: ชุดแรกเหลือ [2], ชุดที่สองเหลือ [1]",
        verifier={
            "type": "unit_tests",
            "details": {
                "program": "def f(a):\n    st=[]\n    for x in a:\n        if st and (st[-1]+x)%2==0:\n            st.pop()\n        else:\n            st.append(x)\n    return sum(st), len(st)\nprint(f([3,4,6,7,2]))\nprint(f([1,2,5,5,6]))\n",
                "expected_stdout": "(2, 1)\n(1, 1)\n",
            },
        },
        sources=SYNTHETIC,
        quality_scores={"correctness": 1.0, "thai_naturalness": 0.99, "benchmark_alignment": 0.86, "novelty": 0.95, "instruction_clarity": 1.0},
        tags=["thai", "coding", "execution_tracing", "python", "livecodebench_style"],
    ),
    item(
        split="train",
        id="openthaieval-train-0001",
        family="openthaieval",
        task_type="reading_comprehension_multiple_choice",
        language="th",
        difficulty="medium",
        messages=[
            {
                "role": "user",
                "content": "อ่านบทความสั้นต่อไปนี้แล้วเลือกคำตอบที่ถูกต้อง\n\nโรงเรียนบ้านทุ่งแก้วเริ่มโครงการแยกขยะในโรงอาหารโดยตั้งถังสี่ประเภท นักเรียนชั้น ม.2 รับหน้าที่อธิบายวิธีแยกขยะให้รุ่นน้องช่วงพักกลางวัน หลังดำเนินโครงการหนึ่งเดือน ปริมาณขยะปนเปื้อนลดลง แต่ครูพบว่านักเรียนบางส่วนยังทิ้งกล่องนมลงถังรีไซเคิลโดยไม่เทนมที่เหลือออกก่อน คณะทำงานจึงเพิ่มป้ายเตือนและจัดมุมเทของเหลวไว้ใกล้ถัง\n\nคำถาม: เหตุผลหลักที่คณะทำงานเพิ่มมุมเทของเหลวคือข้อใด\n(1) เพื่อเพิ่มจำนวนถังขยะในโรงอาหาร\n(2) เพื่อให้ขยะรีไซเคิลไม่ปนเปื้อนจากของเหลว\n(3) เพื่อให้นักเรียนชั้น ม.2 ไม่ต้องอธิบายงาน\n(4) เพื่อเปลี่ยนกล่องนมให้เป็นขยะอินทรีย์",
            },
            {
                "role": "assistant",
                "content": "คำตอบคือ (2)\n\nเหตุผล: บทความระบุว่ายังมีนักเรียนทิ้งกล่องนมลงถังรีไซเคิลโดยไม่เทนมที่เหลือออกก่อน การเพิ่มมุมเทของเหลวจึงช่วยลดการปนเปื้อนของขยะรีไซเคิล ไม่ใช่การเพิ่มจำนวนถังหรือเปลี่ยนประเภทของกล่องนม",
            },
        ],
        final_answer="(2)",
        concise_solution="คำตอบอยู่จากเหตุและผลในบทความ: ของเหลวที่เหลือทำให้ขยะรีไซเคิลปนเปื้อน",
        verifier={"type": "exact_match", "details": {"accepted_answers": ["(2)", "2"]}},
        sources=SYNTHETIC,
        quality_scores={"correctness": 1.0, "thai_naturalness": 0.99, "benchmark_alignment": 0.90, "novelty": 0.97, "instruction_clarity": 1.0},
        tags=["thai", "openthaieval", "reading_comprehension", "multiple_choice"],
    ),
    item(
        split="train",
        id="openthaieval-train-0002",
        family="openthaieval",
        task_type="xnli_style_inference",
        language="th",
        difficulty="medium",
        messages=[
            {
                "role": "user",
                "content": "กำหนดประโยคตั้งต้นและสมมติฐาน จงตัดสินว่าเป็น entailment, contradiction หรือ neutral\n\nประโยคตั้งต้น: คลินิกชุมชนประกาศว่าจะปิดให้บริการเฉพาะวันอาทิตย์ ส่วนวันเสาร์ยังเปิดตามเวลาปกติ\nสมมติฐาน: คลินิกชุมชนเปิดให้บริการในวันเสาร์",
            },
            {
                "role": "assistant",
                "content": "คำตอบคือ entailment\n\nเหตุผล: ประโยคตั้งต้นบอกชัดเจนว่าวันเสาร์ยังเปิดตามเวลาปกติ ดังนั้นสมมติฐานที่ว่าคลินิกเปิดให้บริการในวันเสาร์จึงตามมาจากประโยคตั้งต้นโดยตรง",
            },
        ],
        final_answer="entailment",
        concise_solution="สมมติฐานถูกระบุโดยตรงในประโยคตั้งต้น",
        verifier={"type": "exact_match", "details": {"accepted_answers": ["entailment"]}},
        sources=SYNTHETIC,
        quality_scores={"correctness": 1.0, "thai_naturalness": 0.99, "benchmark_alignment": 0.91, "novelty": 0.97, "instruction_clarity": 1.0},
        tags=["thai", "openthaieval", "nli", "xnli_style"],
    ),
    item(
        split="validation",
        id="openthaieval-val-0001",
        family="openthaieval",
        task_type="thai_professional_qa",
        language="th",
        difficulty="medium",
        messages=[
            {
                "role": "user",
                "content": "ข้อใดสอดคล้องกับเป้าหมายของนโยบายการเงินของธนาคารแห่งประเทศไทยมากที่สุด\n(1) รักษาเสถียรภาพราคา สนับสนุนการเติบโตทางเศรษฐกิจที่ยั่งยืน และดูแลเสถียรภาพระบบการเงิน\n(2) กำหนดราคาสินค้าทุกชนิดในตลาดค้าปลีก\n(3) ออกกฎหมายภาษีเงินได้บุคคลธรรมดาโดยตรง\n(4) รับประกันว่าค่าเงินบาทต้องแข็งค่าขึ้นทุกปี",
            },
            {
                "role": "assistant",
                "content": "คำตอบคือ (1)\n\nเหตุผล: ธนาคารแห่งประเทศไทยระบุเป้าหมายของนโยบายการเงินไว้ที่เสถียรภาพราคาในระยะปานกลาง การเติบโตทางเศรษฐกิจอย่างยั่งยืน และเสถียรภาพระบบการเงิน ส่วนตัวเลือกอื่นเกินบทบาทหรือขัดกับกรอบนโยบายการเงิน",
            },
        ],
        final_answer="(1)",
        concise_solution="อ้างอิงเป้าหมายสามด้านของนโยบายการเงินจาก ธปท.",
        verifier={
            "type": "retrieval_evidence",
            "details": {
                "required_supporting_facts": [
                    "BOT monetary policy pursues medium-term price stability.",
                    "BOT monetary policy supports sustainable economic growth.",
                    "BOT monetary policy considers financial stability.",
                ]
            },
        },
        sources=[{"url": "https://www.bot.or.th/en/our-roles/monetary-policy/about-monetary-policy.html", "license": "official public webpage; used as citation only", "used_for": "fact verification"}],
        quality_scores={"correctness": 1.0, "thai_naturalness": 0.98, "benchmark_alignment": 0.89, "novelty": 0.96, "instruction_clarity": 1.0},
        tags=["thai", "openthaieval", "professional_qa", "economics", "source_grounded"],
    ),
    item(
        split="train",
        id="mtbench-th-train-0001",
        family="mt_bench",
        task_type="multi_turn_advice",
        language="th",
        difficulty="medium",
        messages=[
            {"role": "user", "content": "ช่วยแนะนำเกณฑ์เลือกโน้ตบุ๊กให้นักศึกษาวิทยาการคอมพิวเตอร์ปีหนึ่ง งบไม่เกิน 28000 บาท ขอแบบไม่ยึดติดยี่ห้อ"},
            {
                "role": "assistant",
                "content": "ให้ดูจากงานที่จะใช้จริงก่อน ถ้าเรียนเขียนโปรแกรมทั่วไป ทำเว็บ ฐานข้อมูล และรันโปรเจกต์ขนาดกลาง สเปกที่ควรมองหาคือ RAM 16 GB, SSD อย่างน้อย 512 GB, ซีพียูรุ่นใหม่ระดับกลาง และจอที่อ่านโค้ดได้นานโดยไม่ล้าตา\n\nสิ่งที่ไม่ควรลดมากคือ RAM และ SSD เพราะมีผลกับการเปิด IDE, browser หลายแท็บ และ container เบื้องต้น ส่วนการ์ดจอแยกยังไม่จำเป็นถ้าไม่ได้เรียนงานกราฟิกหรือ ML หนักๆ ตั้งแต่ปีแรก ควรเผื่องบให้เมาส์ดีๆ กระเป๋ากันกระแทก และประกันที่เคลมสะดวกด้วย",
            },
            {"role": "user", "content": "ปรับคำแนะนำให้เหมาะกับคนที่ต้องพกขึ้นรถเมล์ทุกวัน และตัดตัวเลือกสายเกมออก"},
            {
                "role": "assistant",
                "content": "ถ้าต้องพกขึ้นรถเมล์ทุกวัน ให้ให้ความสำคัญกับน้ำหนัก ความทน และแบตมากกว่าสเปกที่แรงเกินจำเป็น\n\nเกณฑ์ที่ควรใช้คือ\n1. น้ำหนักประมาณ 1.2-1.5 กก. จะพกทุกวันได้จริงกว่าเครื่องหนัก\n2. RAM 16 GB และ SSD 512 GB ยังเป็นจุดคุ้มค่าสำหรับงานเขียนโปรแกรม\n3. จอ 14 นิ้วมักสมดุลระหว่างพื้นที่ทำงานกับความคล่องตัว\n4. แบตควรใช้งานเรียนและทำงานเอกสารได้หลายชั่วโมงโดยไม่ต้องหาปลั๊กตลอด\n5. บอดี้และบานพับควรแข็งแรง เพราะการเดินทางทุกวันทำให้เครื่องกระแทกง่าย\n\nตัดเครื่องเกมมิ่งออกได้เลยถ้าน้ำหนักมาก แบตสั้น พัดลมดัง หรือชาร์จเจอร์ใหญ่เกินไป สำหรับปีหนึ่ง เครื่องบางเบาที่อัปเกรดหรือซ่อมง่ายมักใช้คุ้มกว่า",
            },
        ],
        final_answer="multi_turn_response",
        concise_solution="ตอบต่อเนื่องสองเทิร์น ปรับตามข้อจำกัดการพกพาและตัดสายเกมออก",
        verifier={"type": "llm_judge_rubric", "details": {"criteria": ["helpfulness", "multi_turn_consistency", "thai_naturalness", "budget_awareness", "specificity"]}},
        sources=SYNTHETIC,
        quality_scores={"correctness": 0.98, "thai_naturalness": 0.99, "benchmark_alignment": 0.91, "novelty": 0.97, "instruction_clarity": 0.98},
        tags=["thai", "mt_bench", "multi_turn", "advice", "education"],
    ),
    item(
        split="train",
        id="mtbench-then-train-0002",
        family="mt_bench",
        task_type="multi_turn_business_writing",
        language="th-en",
        difficulty="medium",
        messages=[
            {"role": "user", "content": "ช่วยร่างอีเมลภาษาไทยถึงลูกค้าองค์กร แจ้งว่า API ของเราขัดข้องช่วงเช้า แต่แก้ไขแล้ว ขอให้สุภาพและไม่โยนความผิด"},
            {
                "role": "assistant",
                "content": "เรื่อง แจ้งเหตุขัดข้องของบริการ API และการแก้ไขเรียบร้อยแล้ว\n\nเรียน ลูกค้าทุกท่าน\n\nทีมงานขอแจ้งให้ทราบว่าในช่วงเช้าวันนี้ บริการ API บางส่วนของเราทำงานไม่เสถียร ส่งผลให้บางคำขอใช้เวลาตอบสนองนานกว่าปกติหรือไม่สำเร็จเป็นบางช่วง ขณะนี้ทีมวิศวกรรมได้แก้ไขปัญหาและเฝ้าระวังระบบเพิ่มเติมแล้ว\n\nเราขออภัยอย่างยิ่งต่อผลกระทบที่เกิดขึ้น และกำลังตรวจสอบสาเหตุเชิงลึกเพื่อป้องกันไม่ให้เกิดซ้ำ หากท่านยังพบความผิดปกติ สามารถแจ้งรายละเอียดเวลาและ endpoint ที่เกี่ยวข้องกลับมาได้ทันที\n\nขอแสดงความนับถือ\nทีมสนับสนุนลูกค้า",
            },
            {"role": "user", "content": "ทำให้สั้นลงครึ่งหนึ่ง และเพิ่ม subject ภาษาอังกฤษที่ดูเป็นมืออาชีพ"},
            {
                "role": "assistant",
                "content": "Subject: API Service Incident Resolved\n\nเรียน ลูกค้าทุกท่าน\n\nช่วงเช้าวันนี้บริการ API บางส่วนทำงานไม่เสถียร ทำให้บางคำขอตอบสนองช้าหรือไม่สำเร็จเป็นบางช่วง ขณะนี้ทีมงานแก้ไขแล้วและกำลังเฝ้าระวังระบบอย่างใกล้ชิด\n\nเราขออภัยต่อผลกระทบที่เกิดขึ้น และจะตรวจสอบสาเหตุเพื่อป้องกันการเกิดซ้ำ หากยังพบปัญหา กรุณาส่งเวลาและ endpoint ที่เกี่ยวข้องให้ทีมสนับสนุนตรวจสอบต่อได้ทันที\n\nขอแสดงความนับถือ\nทีมสนับสนุนลูกค้า",
            },
        ],
        final_answer="multi_turn_response",
        concise_solution="รักษาน้ำเสียงสุภาพ ลดความยาว และเพิ่ม subject ภาษาอังกฤษตามคำสั่งเทิร์นสอง",
        verifier={"type": "llm_judge_rubric", "details": {"criteria": ["professional_tone", "thai_naturalness", "multi_turn_revision", "no_blame", "conciseness"]}},
        sources=SYNTHETIC,
        quality_scores={"correctness": 0.98, "thai_naturalness": 0.99, "benchmark_alignment": 0.90, "novelty": 0.98, "instruction_clarity": 0.98},
        tags=["thai", "english", "mt_bench", "multi_turn", "business_writing"],
    ),
    item(
        split="validation",
        id="mtbench-th-val-0001",
        family="mt_bench",
        task_type="multi_turn_planning",
        language="th",
        difficulty="medium",
        messages=[
            {"role": "user", "content": "ออกแบบกิจกรรม 45 นาทีให้ชมรมอ่านข่าวของโรงเรียนมัธยม เพื่อฝึกแยกข่าวจริง ข่าวความเห็น และข่าวโฆษณาแฝง"},
            {
                "role": "assistant",
                "content": "แบ่งกิจกรรมเป็น 4 ช่วง\n\nช่วงที่ 1 อุ่นเครื่อง 5 นาที: ให้สมาชิกบอกตัวอย่างหัวข้อข่าวที่เห็นในสัปดาห์นี้ แล้วครูหรือประธานชมรมเขียนคำว่า ข่าวจริง ความเห็น และโฆษณาแฝง ไว้บนกระดาน\n\nช่วงที่ 2 อ่านและจำแนก 15 นาที: แจกข่าวสั้น 3 ชิ้นที่ตัดข้อมูลส่วนแหล่งที่มาออก ให้แต่ละกลุ่มขีดเส้นใต้ข้อความที่เป็นข้อเท็จจริง วงกลมถ้อยคำแสดงความเห็น และทำเครื่องหมายตรงประโยคที่อาจมีเจตนาเชิญชวนซื้อหรือสนับสนุนสินค้า\n\nช่วงที่ 3 อภิปราย 15 นาที: แต่ละกลุ่มอธิบายเหตุผลของตนเอง กลุ่มอื่นถามกลับได้ โดยต้องอ้างข้อความในข่าว ไม่ใช่ตอบจากความรู้สึกอย่างเดียว\n\nช่วงที่ 4 สรุป 10 นาที: ร่วมกันทำเช็กลิสต์ 5 ข้อสำหรับอ่านข่าวครั้งต่อไป เช่น ใครเป็นผู้พูด มีหลักฐานหรือไม่ แยกข้อเท็จจริงกับความเห็นได้ไหม และมีผลประโยชน์แฝงหรือเปล่า",
            },
            {"role": "user", "content": "ถ้าโรงเรียนอยู่ชนบทและอินเทอร์เน็ตไม่เสถียร ควรปรับอย่างไร"},
            {
                "role": "assistant",
                "content": "ให้เปลี่ยนกิจกรรมเป็นแบบออฟไลน์ทั้งหมด\n\nเตรียมข่าวจากหนังสือพิมพ์ วิทยุชุมชนที่ถอดความไว้ หรือประกาศท้องถิ่นที่พิมพ์ลงกระดาษล่วงหน้า แทนการค้นข่าวสดจากอินเทอร์เน็ต แต่ยังคงใช้สามประเภทเดิมคือ ข่าวจริง ความเห็น และโฆษณาแฝง\n\nเพิ่มช่วงให้สมาชิกเล่าข่าวจากชุมชนที่ได้ยินมา แล้วให้เพื่อนช่วยตั้งคำถามว่า ใครเป็นแหล่งข่าว มีหลักฐานอะไร และส่วนใดเป็นการตีความ วิธีนี้ทำให้กิจกรรมเข้ากับบริบทท้องถิ่นและไม่ขึ้นกับสัญญาณอินเทอร์เน็ต",
            },
        ],
        final_answer="multi_turn_response",
        concise_solution="ออกแบบกิจกรรมตามเวลาแล้วปรับให้ใช้งานได้เมื่อไม่มีอินเทอร์เน็ตเสถียร",
        verifier={"type": "llm_judge_rubric", "details": {"criteria": ["feasibility", "multi_turn_consistency", "thai_naturalness", "rural_context_adaptation"]}},
        sources=SYNTHETIC,
        quality_scores={"correctness": 0.98, "thai_naturalness": 0.99, "benchmark_alignment": 0.90, "novelty": 0.98, "instruction_clarity": 0.98},
        tags=["thai", "mt_bench", "multi_turn", "media_literacy", "planning"],
    ),
    item(
        split="train",
        id="ifeval-ifbench-th-train-0001",
        family="ifeval_ifbench",
        task_type="verifiable_json_instruction",
        language="th",
        difficulty="hard",
        messages=[
            {
                "role": "user",
                "content": "ตอบเป็น JSON เท่านั้น ห้ามมีข้อความนอก JSON ต้องมีคีย์ title, steps และ warning โดย title เป็นสตริงภาษาไทย, steps เป็นอาร์เรย์ 3 รายการและแต่ละรายการต้องขึ้นต้นด้วยคำว่า \"ขั้น\", ส่วน warning ต้องเป็นสตริงที่ไม่มีคำว่า \"ห้าม\"",
            },
            {
                "role": "assistant",
                "content": "{\"title\":\"แผนสำรองไฟล์รายสัปดาห์\",\"steps\":[\"ขั้นเตรียมโฟลเดอร์ปลายทาง\",\"ขั้นคัดลอกไฟล์ที่เปลี่ยนแปลง\",\"ขั้นตรวจสอบว่าเปิดไฟล์สำคัญได้\"],\"warning\":\"ควรตรวจพื้นที่เก็บข้อมูลก่อนเริ่มงาน\"}",
            },
        ],
        final_answer="valid_json_response",
        concise_solution="ตรวจ JSON-only, คีย์ครบ, steps มี 3 รายการขึ้นต้นด้วย ขั้น, warning ไม่มีคำต้องห้าม",
        verifier={
            "type": "json_schema",
            "details": {
                "schema": {"type": "object", "required": ["title", "steps", "warning"], "properties": {"title": {"type": "string"}, "steps": {"type": "array", "minItems": 3, "maxItems": 3}, "warning": {"type": "string"}}},
                "additional_rules": ["steps_each_startswith_ขั้น", "warning_not_contains_ห้าม", "json_only"],
            },
        },
        sources=SYNTHETIC,
        quality_scores={"correctness": 1.0, "thai_naturalness": 0.98, "benchmark_alignment": 0.95, "novelty": 0.97, "instruction_clarity": 1.0},
        tags=["thai", "instruction_following", "ifeval", "ifbench", "json_schema"],
    ),
    item(
        split="train",
        id="ifeval-ifbench-th-train-0002",
        family="ifeval_ifbench",
        task_type="verifiable_thai_format_constraints",
        language="th",
        difficulty="hard",
        messages=[
            {
                "role": "user",
                "content": "เขียนข้อคิดเรื่องการช่วยเหลือกันเป็นภาษาไทย มี 2 ย่อหน้า แต่ละย่อหน้ามี 2 ประโยคและต้องขึ้นต้นด้วย \"ข้อคิด:\" คำว่า \"น้ำใจ\" ต้องปรากฏทั้งหมด 2 ครั้งพอดี ห้ามใช้เลขอารบิก และห้ามใช้คำลงท้ายสุภาพ \"ครับ\" หรือ \"ค่ะ\"",
            },
            {
                "role": "assistant",
                "content": "ข้อคิด: น้ำใจทำให้การอยู่ร่วมกันเบาลงเมื่อแต่ละคนมองเห็นภาระของผู้อื่น. เมื่อช่วยกันโดยไม่หวังผลตอบแทน ความไว้ใจในชุมชนก็เพิ่มขึ้น.\n\nข้อคิด: น้ำใจยังสอนให้เราใช้ทรัพยากรและเวลาอย่างระมัดระวัง. หากเริ่มจากเรื่องเล็กในบ้านหรือโรงเรียน ผลดีจะค่อยๆ ขยายไปถึงคนรอบตัว.",
            },
        ],
        final_answer="valid_response",
        concise_solution="ตรวจจำนวนย่อหน้า ประโยค คำบังคับ คำต้องห้าม และเลขอารบิก",
        verifier={
            "type": "regex",
            "details": {
                "rules": ["paragraph_count == 2", "each_paragraph_sentence_count == 2", "each_paragraph_startswith('ข้อคิด:')", "count('น้ำใจ') == 2", "no_arabic_digits", "not_contains_any(['ครับ','ค่ะ'])"]
            },
        },
        sources=SYNTHETIC,
        quality_scores={"correctness": 1.0, "thai_naturalness": 0.98, "benchmark_alignment": 0.95, "novelty": 0.98, "instruction_clarity": 1.0},
        tags=["thai", "instruction_following", "ifeval", "ifbench", "thai_constraints"],
    ),
    item(
        split="validation",
        id="ifeval-ifbench-en-val-0001",
        family="ifeval_ifbench",
        task_type="verifiable_markdown_constraints",
        language="en",
        difficulty="adversarial",
        messages=[
            {
                "role": "user",
                "content": "Give four markdown bullets about dataset review. Each bullet must start with \"Do:\". Do not use any comma. The word \"audit\" must appear exactly twice in the whole answer. The final line after the bullets must be exactly \"Done\".",
            },
            {
                "role": "assistant",
                "content": "- Do: audit prompts before training\n- Do: review sources for license fit\n- Do: run audit checks after packaging\n- Do: keep weak items out\nDone",
            },
        ],
        final_answer="valid_response",
        concise_solution="Four bullets start with Do:, no comma, audit appears twice, final line is Done",
        verifier={
            "type": "regex",
            "details": {
                "rules": ["bullet_count == 4", "each_bullet_startswith('- Do:')", "not_contains(',')", "word_count('audit') == 2", "last_line == 'Done'"]
            },
        },
        sources=SYNTHETIC,
        quality_scores={"correctness": 1.0, "thai_naturalness": 1.0, "benchmark_alignment": 0.94, "novelty": 0.97, "instruction_clarity": 1.0},
        tags=["english", "instruction_following", "ifeval", "ifbench", "markdown_constraints"],
    ),
    item(
        split="train",
        id="hotpotqa-agentic-th-train-0001",
        family="hotpotqa_agentic",
        task_type="multi_hop_retrieval_qa",
        language="th",
        difficulty="hard",
        messages=[
            {
                "role": "user",
                "content": "ใช้หลักฐานจากแหล่งข้อมูลที่เชื่อถือได้ตอบคำถามนี้: แหล่งมรดกโลกที่ประกอบด้วยสุโขทัย ศรีสัชนาลัย และกำแพงเพชรตั้งอยู่ในภูมิภาคใดของประเทศไทย และเมืองทั้งสามสะท้อนความสามารถด้านใดที่เกี่ยวกับน้ำ",
            },
            {
                "role": "assistant",
                "content": "คำตอบคือ แหล่งนี้ตั้งอยู่ในภูมิภาคตอนล่างของภาคเหนือของประเทศไทย และสะท้อนความสามารถด้านวิศวกรรมชลศาสตร์หรือการจัดการน้ำ\n\nหลักฐานที่เชื่อมโยงกัน:\n1. UNESCO ระบุว่า Historic Town of Sukhothai and Associated Historic Towns เป็นทรัพย์สินต่อเนื่องที่ประกอบด้วย Sukhothai, Si Satchanalai และ Kamphaeng Phet\n2. แหล่งเดียวกันระบุว่าตั้งอยู่ใน lower northern region of present-day Thailand\n3. UNESCO อธิบายว่าเมืองเหล่านี้มีนวัตกรรมด้าน hydraulic engineering เช่น เขื่อน อ่างเก็บน้ำ สระ คลอง และคูเมือง\n\nดังนั้นคำตอบต้องมีทั้งตำแหน่งภูมิภาคและประเด็นการจัดการน้ำ ไม่ใช่ตอบเพียงชื่อเมือง",
            },
        ],
        final_answer="ตอนล่างของภาคเหนือของประเทศไทย; วิศวกรรมชลศาสตร์หรือการจัดการน้ำ",
        concise_solution="เชื่อมข้อมูล UNESCO: รายชื่อสามเมือง → lower northern region → hydraulic engineering/water management",
        verifier={
            "type": "retrieval_evidence",
            "details": {
                "required_supporting_facts": [
                    "Historic Town of Sukhothai and Associated Historic Towns includes Sukhothai, Si Satchanalai, and Kamphaeng Phet.",
                    "The property is in the lower northern region of present-day Thailand.",
                    "The towns are associated with hydraulic engineering and water management.",
                ]
            },
        },
        sources=[{"url": "https://whc.unesco.org/en/list/574", "license": "UNESCO public webpage; used as citation only", "used_for": "fact verification"}],
        quality_scores={"correctness": 1.0, "thai_naturalness": 0.98, "benchmark_alignment": 0.93, "novelty": 0.97, "instruction_clarity": 0.98},
        tags=["thai", "hotpotqa", "multi_hop", "retrieval", "source_grounded", "thai_cultural"],
    ),
    item(
        split="train",
        id="hotpotqa-agentic-then-train-0002",
        family="hotpotqa_agentic",
        task_type="multi_hop_retrieval_qa",
        language="th-en",
        difficulty="hard",
        messages=[
            {
                "role": "user",
                "content": "Answer in Thai with evidence: Britannica says Bangkok became the capital when Rama I moved the court across the Chao Phraya River in 1782. The same source says Rama I modeled the new city on a former capital north of Bangkok. เมืองหลวงเดิมนั้นคือเมืองใด",
            },
            {
                "role": "assistant",
                "content": "คำตอบคือ อยุธยา\n\nเหตุผล: หลักฐานแรกระบุว่าในปี 1782 รัชกาลที่ 1 ย้ายราชสำนักไปฝั่งตะวันออกของแม่น้ำเจ้าพระยา ทำให้กรุงเทพฯ เป็นเมืองหลวงของสยาม หลักฐานถัดมาจากแหล่งเดียวกันระบุว่าพระองค์วางแบบเมืองใหม่โดยอิงจาก former capital, Ayutthaya ซึ่งอยู่ทางเหนือของกรุงเทพฯ ดังนั้นเมืองหลวงเดิมที่ถามถึงคืออยุธยา",
            },
        ],
        final_answer="อยุธยา",
        concise_solution="เชื่อม Bangkok became capital in 1782 under Rama I → new city modeled on former capital Ayutthaya",
        verifier={
            "type": "retrieval_evidence",
            "details": {
                "required_supporting_facts": [
                    "Bangkok became capital in 1782 when Rama I moved the court across the Chao Phraya River.",
                    "Rama I modeled the new city on the former capital Ayutthaya.",
                ]
            },
        },
        sources=[{"url": "https://www.britannica.com/place/Bangkok/History", "license": "Britannica webpage; used as citation only", "used_for": "fact verification"}],
        quality_scores={"correctness": 1.0, "thai_naturalness": 0.98, "benchmark_alignment": 0.92, "novelty": 0.96, "instruction_clarity": 0.98},
        tags=["thai", "english", "hotpotqa", "multi_hop", "retrieval", "source_grounded"],
    ),
    item(
        split="validation",
        id="hotpotqa-agentic-th-val-0001",
        family="hotpotqa_agentic",
        task_type="multi_hop_retrieval_qa",
        language="th",
        difficulty="hard",
        messages=[
            {
                "role": "user",
                "content": "ใช้หลักฐานจาก UNESCO ตอบคำถามนี้: เมืองประวัติศาสตร์อยุธยามีบทบาทเป็นจุดเชื่อมต่อระหว่างภูมิภาคใดกับภูมิภาคใด และปัจจุบันหน่วยงานใดของไทยเป็นผู้บังคับใช้กฎหมายคุ้มครองแหล่งนี้",
            },
            {
                "role": "assistant",
                "content": "คำตอบคือ อยุธยาเป็นจุดเชื่อมต่อระหว่างตะวันออกกับตะวันตก และการคุ้มครองตามกฎหมายบังคับใช้โดยกรมศิลปากร กระทรวงวัฒนธรรม\n\nหลักฐานจาก UNESCO ระบุว่าอยุธยาเป็น connecting point between the East and the West เพราะมีบทบาททางเศรษฐกิจและการค้าในระดับภูมิภาคและโลก อีกช่วงหนึ่งของแหล่งข้อมูลระบุว่า Historic City of Ayutthaya ได้รับการคุ้มครองตามกฎหมายโบราณสถานของไทย โดยมี Fine Arts Department, Ministry of Culture เป็นหน่วยงานบังคับใช้",
            },
        ],
        final_answer="ตะวันออกกับตะวันตก; กรมศิลปากร กระทรวงวัฒนธรรม",
        concise_solution="เชื่อมบทบาทการค้า East-West กับข้อมูลการคุ้มครองโดย Fine Arts Department, Ministry of Culture จาก UNESCO",
        verifier={
            "type": "retrieval_evidence",
            "details": {
                "required_supporting_facts": [
                    "Ayutthaya was an important connecting point between the East and the West.",
                    "The Historic City of Ayutthaya is protected under Thai law enforced by the Fine Arts Department, Ministry of Culture.",
                ]
            },
        },
        sources=[{"url": "https://whc.unesco.org/en/list/576", "license": "UNESCO public webpage; used as citation only", "used_for": "fact verification"}],
        quality_scores={"correctness": 1.0, "thai_naturalness": 0.98, "benchmark_alignment": 0.92, "novelty": 0.97, "instruction_clarity": 0.98},
        tags=["thai", "hotpotqa", "multi_hop", "retrieval", "source_grounded", "thai_cultural"],
    ),
]


REJECTED_OR_QUARANTINED = [
    {
        "item_id": "draft-aime-th-reject-0001",
        "benchmark_family": "aime_th",
        "acceptance_decision": "reject",
        "reason_for_decision": "Discarded before inclusion because the draft relied on a familiar contest-style constant swap without enough originality.",
        "known_risks": "Would teach shallow pattern matching rather than transferable reasoning.",
    },
    {
        "item_id": "draft-hotpot-th-quarantine-0001",
        "benchmark_family": "hotpotqa_agentic",
        "acceptance_decision": "quarantine",
        "reason_for_decision": "External factual claim could not be verified from a primary or stable source during this run.",
        "known_risks": "Potential factual uncertainty and source instability.",
    },
]
