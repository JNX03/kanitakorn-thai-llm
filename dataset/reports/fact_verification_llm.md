# Fact verification report (LLM-as-judge)

Judge: codex (gpt-5.5 xhigh). Backend reads the full fetched HTML excerpt and decides whether sources actually support the claims.

**OK: 25 · WARN: 3 · FAIL: 2 · TOTAL: 30**

| id | verdict | claimed answer | rationale | unsupported facts |
|---|---|---|---|---|
| hotpotqa-agentic-th-seed-0000 | OK | จังหวัดสิงห์บุรี | Both required facts are clearly stated in the provided source excerpts. | — |
| hotpotqa-agentic-th-seed-0001 | WARN | พระสหาย | Fact 0 is explicit, but fact 1 is only tangentially implied by a St. Petersburg photo caption and is not clearly stated. | 1 |
| hotpotqa-agentic-th-seed-0002 | OK | ประเทศไทย | Both facts are clearly stated in the provided excerpts. | — |
| hotpotqa-agentic-th-seed-0003 | OK | 1982 | Sources clearly state that Love Destiny/Bupphe Sanniwat is a 2018 series starring Thanavat Vatthanaputi and that he was born on 27 December 1982. | — |
| hotpotqa-agentic-th-seed-0004 | OK | 3,776 เมตร | Both sources clearly state the 1964 Summer Olympics were hosted in Tokyo, Japan, and that Mount Fuji is Japan's highest mountain at about 3,776 meters | — |
| hotpotqa-agentic-th-seed-0005 | OK | อิตาลี | Both sources clearly support Ferrari's prancing horse emblem and Enzo Ferrari being the Ferrari founder born in Italy. | — |
| hotpotqa-agentic-th-seed-0006 | FAIL | รัชกาลที่ 1 | แหล่งข้อมูลระบุว่าวัดพระแก้วโปรดให้สร้างโดยรัชกาลที่ 1 เมื่อ พ.ศ. 2325 ไม่ใช่เริ่ม พ.ศ. 2326 ส่วนแหล่งรัชกาลที่ 1 ระบุพระองค์เป็นรัชกาลที่ 1 แห่งราชวง | 0 |
| hotpotqa-agentic-th-seed-0007 | OK | รัสเซีย | Both required facts are clearly stated in the provided source excerpts. | — |
| hotpotqa-agentic-th-seed-0008 | OK | 1883 | Both sources clearly state that พระเจนดุริยางค์ composed the current Thai national anthem melody and was born in พ.ศ. 2426, equivalent to 1883 CE. | — |
| hotpotqa-agentic-th-seed-0009 | OK | นครนิวยอร์ก สหรัฐอเมริกา | Sources clearly state the Statue of Liberty is on Liberty Island, part of New York City, U.S., in New York Harbor. | — |
| hotpotqa-agentic-th-seed-0010 | OK | ภาษาฝรั่งเศส | แหล่งข้อมูลระบุว่าหอไอเฟลอยู่ที่ปารีส ประเทศฝรั่งเศส และฝรั่งเศสมีภาษาฝรั่งเศสเป็นภาษาทางการ | — |
| hotpotqa-agentic-th-seed-0011 | WARN | สงครามโลกครั้งที่ 2 | Fact 0 is clearly stated, but fact 1 is only inferable from reconnaissance in 1944 and a German shootdown, not explicitly stated as World War II. | 1 |
| hotpotqa-agentic-th-seed-0012 | OK | อังกฤษ | แหล่งข้อมูลระบุว่ากฎนี้เป็นส่วนสำคัญในงาน Principia ของนิวตันที่เผยแพร่ครั้งแรก และระบุว่าไอแซก นิวตันเกิดในราชอาณาจักรอังกฤษ | — |
| hotpotqa-agentic-th-seed-0013 | OK | ฟิสิกส์ | Both required facts are clearly stated in the provided source excerpts. | — |
| hotpotqa-agentic-th-seed-0014 | OK | ทวีปยุโรปและเอเชีย | Both sources state the Caspian Sea is an inland/endorheic closed water body and the largest such lake/body of water, and the Thai source explicitly pl | — |
| hotpotqa-agentic-th-seed-0015 | OK | ฟินแลนด์ ปี 1991 | Both sources clearly state that Linux was created/released by Linus Torvalds and that Torvalds is Finnish and has led/created the Linux kernel since 1 | — |
| hotpotqa-agentic-th-seed-0016 | OK | กัมพูชา ศตวรรษที่ 12 | แหล่งข้อมูลระบุชัดว่านครวัดอยู่ในกัมพูชา เป็นศาสนสถานที่ใหญ่ที่สุดในโลก และสร้างช่วงต้นคริสต์ศตวรรษที่ 12 ในรัชสมัยพระเจ้าสุริยวรมันที่ 2 | — |
| hotpotqa-agentic-th-seed-0017 | OK | เกาะภูเก็ต | Both facts are clearly stated in the provided source excerpts. | — |
| hotpotqa-agentic-th-seed-0018 | OK | 1948 | Both sources clearly state that WHO is headquartered in Geneva, Switzerland, and was founded in 1948. | — |
| hotpotqa-agentic-th-seed-0019 | OK | 1980 | The excerpts state that Imagine is a single/song by John Lennon and that John Lennon died on 8 December 1980. | — |
| hotpotqa-agentic-th-seed-0020 | OK | ภาษาเยอรมัน | แหล่งข้อมูลระบุว่า Lindt เป็นบริษัทช็อกโกแลตสวิส และสวิตเซอร์แลนด์มีภาษาเยอรมันเป็นภาษาทางการที่มีสัดส่วนสูงสุด 62%. | — |
| hotpotqa-agentic-th-seed-0021 | OK | 1945 | Both sources clearly state that penicillin was discovered by Alexander Fleming and that he shared the Nobel Prize in Physiology or Medicine in 1945. | — |
| hotpotqa-agentic-th-seed-0022 | OK | 25 กม., พ.ศ. 2549 | แหล่งข้อมูลระบุว่าสนามบินสุวรรณภูมิเปิดเมื่อ 28 กันยายน 2006 ซึ่งตรงกับ พ.ศ. 2549 และอยู่ห่างจากใจกลางเมืองประมาณ 25 กิโลเมตร | — |
| hotpotqa-agentic-th-seed-0023 | OK | รัชกาลที่ 1, พ.ศ. 2325 | แหล่งข้อมูลระบุชัดว่าพระบรมมหาราชวังโปรดเกล้าฯ ให้สร้างในรัชสมัยรัชกาลที่ 1 และเริ่มสร้างเมื่อวันที่ 6 พฤษภาคม พ.ศ. 2325 | — |
| hotpotqa-agentic-th-seed-0024 | OK | สัญชาติอังกฤษ (British) | Both source excerpts clearly state Rowling was a British author and Conan Doyle was a British writer born in Edinburgh, Scotland. | — |
| hotpotqa-agentic-th-seed-0025 | OK | Washington, สหรัฐอเมริกา | The excerpts clearly state that Bill Gates co-founded Microsoft and was born in Seattle, Washington, U.S. | — |
| hotpotqa-agentic-th-seed-0026 | WARN | เยอรมนี | Fact 0 is supported by combining the deafness timeline with Symphony No. 9 composition dates, but the sources state Bonn and German rather than clearl | 1 |
| hotpotqa-agentic-th-seed-0027 | OK | Feijoada | Both sources clearly support that Rio Carnival is in Brazil and that feijoada is Brazil's national dish. | — |
| hotpotqa-agentic-th-seed-0028 | FAIL | เดอะ บิลเลียนแนร์ | Fact 0 is directly stated, but the sources list the 2554 film as 'Top Secret วัยรุ่นพันล้าน' and do not clearly state the title 'เดอะ บิลเลียนแนร์'. | 1 |
| hotpotqa-agentic-th-seed-0029 | OK | Apple I (1976), Steve Jobs | Both facts are clearly stated: Apple I was released in July 1976, and Steve Jobs, Apple co-founder, died on October 5, 2011. | — |