"""Re-score IFEval by stripping <think>...</think> from responses."""
import json, sys
sys.path.insert(0, "/root/kanitakorn")
from eval_ifeval_mtbench import check_constraint, load_ifeval_th

d = json.load(open("/root/kanitakorn/reports/qwen3_ifeval_th.json"))
items = d["items"]
all_items = load_ifeval_th()
# Truncated prompts in saved items, so match by prefix
constraints_by_prompt = {a[0][:200]: a[1] for a in all_items}
print("total items:", len(items), "constraints loaded:", len(constraints_by_prompt))

def strip_think(t):
    if "</think>" in t:
        t = t.split("</think>", 1)[1]
    return t.strip()

passed_old = 0
passed_new = 0
scored = 0
for i in items:
    pkey = i.get("prompt", "")[:200]
    constraints = constraints_by_prompt.get(pkey)
    if not constraints:
        continue
    scored += 1
    resp = i.get("response", "")
    answer = strip_think(resp)
    new_sat = all(check_constraint(answer, iid, kw) for iid, kw in constraints)
    if i["satisfied"]: passed_old += 1
    if new_sat: passed_new += 1

print(f"Scored: {scored}")
print(f"Old (with think):  {passed_old}/{scored} = {passed_old/max(scored,1)*100:.1f}%")
print(f"New (strip think): {passed_new}/{scored} = {passed_new/max(scored,1)*100:.1f}%")
