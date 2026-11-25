"""冒烟测试：单题跑通 检索 → 上下文 → 作答（验证包安装与索引可用）

用法：python scripts/smoke_test.py "What were Lowe's cash dividend payments in 2023?"
"""
import os, sys, json, re

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from config import P, load_env

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_OFFLINE", os.environ.get("RAG_OFFLINE", "1"))
import requests
from src.rag import retriever as fr
from src.generate.prompts import ANSWER_SYS

q = sys.argv[1] if len(sys.argv) > 1 else "What was Lowe's cash dividend payments in 2023?"

items = fr.main_path([q])
ctext = fr.load_texts({c for it in items for c in it["cands"]})
items = fr.rerank_all(items, ctext)
ptext = fr.load_texts({p for it in items for p in it["parents"]})
it = items[0]
ctx = fr.build_ctx(it, {**ctext, **ptext})

print(f"路由文档 : {it['doc']}")
print(f"父段落   : {it['parents']}")
print(f"表旁路   : {len(it['facts'])} 条取数结果")
print(f"上下文   : {len(ctx):,} 字符\n")

r = requests.post(fr.BASE + "/chat/completions", timeout=180,
                  headers={"Authorization": "Bearer " + fr.KEY},
                  json={"model": fr.MODEL, "temperature": 0,
                        "messages": [{"role": "user", "content":
                                      ANSWER_SYS + "\n\n" + ctx[:14000] + "\n\n问题：" + q}]})
r.raise_for_status()
out = r.json()["choices"][0]["message"]["content"]
m = re.search(r"\{.*\}", out, re.S)
obj = json.loads(m.group(0)) if m else {"answer": out}
print("可答性:", obj.get("answerable"), "| reason:", obj.get("reason"))
print("facts :", len(obj.get("facts") or []), "条")
print("答案  :", str(obj.get("answer"))[:400])
