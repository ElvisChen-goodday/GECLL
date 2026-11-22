"""缓存 child → doc 索引（供 doc 内向量检索用）"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
from config import P, DATA_DIR, ENV_FILE, load_env
import json
import numpy as np


docs, didx = [], {}
doc_of = []
with open(P("chunks_struct.jsonl"), encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        if r["level"] != "child":
            continue
        d = r["doc_id"]
        if d not in didx:
            didx[d] = len(docs)
            docs.append(d)
        doc_of.append(didx[d])

np.save("child_doc_idx.npy", np.array(doc_of, dtype=np.int16))
json.dump(docs, open(P("child_docs.json"), "w"))
print(f"child {len(doc_of):,} | doc {len(docs):,} | 每 doc 平均 {len(doc_of)/len(docs):.0f} 个 child")
