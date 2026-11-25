"""合并最终判分：verdicts_full（全库基线）+ verdicts_707_dense（表题改进版）→ verdicts_final.jsonl

报告阶段优先读 verdicts_final.jsonl，因此跑完这条就能看到最终成绩。
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from config import P

import json, os
from collections import Counter


def load(path):
    out = {}
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            try:
                r = json.loads(line)
            except Exception:
                continue
            out[r["qid"]] = r
    return out


base = load(P("verdicts_full.jsonl"))
better = load(P("verdicts_707_dense.jsonl"))
if not base:
    raise SystemExit("缺少 verdicts_full.jsonl，先跑 run_full.py --phase judge")
if not better:
    raise SystemExit("缺少 verdicts_707_dense.jsonl，先跑 render_tables_707.py + eval_table_ab.py")

# 补回 tbl 标记（分层报告需要），来源是上下文文件
tbl_of, cat_of = {}, {}
if os.path.exists(P("ctx_all.jsonl")):
    for line in open(P("ctx_all.jsonl"), encoding="utf-8"):
        try:
            c = json.loads(line)
        except Exception:
            continue
        tbl_of[c["qid"]] = bool(c.get("tbl"))
        cat_of[c["qid"]] = c.get("category")

merged = dict(base)
merged.update(better)
for qid, r in merged.items():
    r["tbl"] = tbl_of.get(qid, r.get("tbl", False))
    r["category"] = cat_of.get(qid, r.get("category"))
out = P("verdicts_final.jsonl")
with open(out, "w", encoding="utf-8") as f:
    for qid, r in merged.items():
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

n = len(merged)
c = Counter(r["verdict"] for r in merged.values())
print(f"合并 {len(base)} + {len(better)} → {n} 题  →  {os.path.basename(out)}")
print(f"  correct  {c['correct']:>5} = {c['correct']/n:.1%}")
print(f"  partial  {c['partial']:>5} = {c['partial']/n:.1%}   (correct+partial = {(c['correct']+c['partial'])/n:.1%})")
print(f"  wrong    {c['wrong']:>5} = {c['wrong']/n:.1%}")
print(f"  refused  {c['refused']:>5} = {c['refused']/n:.1%}")
