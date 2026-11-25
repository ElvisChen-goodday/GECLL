"""以 SQLite 为唯一真源：重建表指纹 → 向量 + BM25（key = table_pk）"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
from config import P, DATA_DIR, ENV_FILE, load_env
import os, json, sqlite3, time
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_OFFLINE", os.environ.get("RAG_OFFLINE", "1"))

import numpy as np
import scipy.sparse as sp
from collections import Counter
from FlagEmbedding import BGEM3FlagModel

import re


TOKEN = re.compile(r"[a-z0-9][a-z0-9,\.\-]*")

con = sqlite3.connect(P("finder_tables.db"))
cur = con.cursor()
docs = {r[0]: r for r in cur.execute("select doc_id, ticker, company, fiscal_year from docs")}

# 每张表的行标签（取前 12 个非数字）与年份集合
labels, years = {}, {}
for pk, lab, yr in cur.execute("""select table_pk, row_label, year from cells
                                  where row_label<>'' order by table_pk, row, col"""):
    L = labels.setdefault(pk, [])
    if lab and len(lab) <= 90 and lab not in L and len(L) < 12 and not re.fullmatch(r"[\d\.,\-\(\)\$%\s]+", lab):
        L.append(lab)
    if yr:
        years.setdefault(pk, set()).add(yr)

recs = []
for pk, doc_id, cap, nrows in cur.execute(
        "select table_pk, doc_id, caption, n_rows from tables"):
    d = docs.get(doc_id, (doc_id, doc_id, doc_id, ""))
    lab = labels.get(pk, [])
    yr = sorted(years.get(pk, set()), reverse=True)
    fp = " | ".join(x for x in [d[2], d[1], (cap or "")[:200],
                                "年份: " + ",".join(yr) if yr else "",
                                " ; ".join(lab)] if x)[:1200]
    recs.append({"table_pk": pk, "doc_id": doc_id, "ticker": d[1], "company": d[2],
                 "caption": cap or "", "labels": lab, "years": yr, "fingerprint": fp})
print(f"表 {len(recs):,}")

with open(P("tables_db.jsonl"), "w", encoding="utf-8") as f:
    for r in recs:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

t0 = time.time()
model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True, device=os.environ.get("RAG_DEVICE", "cuda"))
n = len(recs)
vec = np.lib.format.open_memmap("tables_db_vec.npy", mode="w+", dtype=np.float16, shape=(n, 1024))
step = 2000
for i in range(0, n, step):
    out = model.encode([r["fingerprint"] for r in recs[i:i + step]], batch_size=32, max_length=512)["dense_vecs"]
    vec[i:i + step] = np.asarray(out, dtype=np.float16)
    vec.flush()
print(f"向量完成 {time.time()-t0:.1f}s")

def tok(t):
    return [x.strip(".,-").replace(",", "") for x in TOKEN.findall(t.lower()) if len(x) > 1]

docs_t = [tok(r["fingerprint"]) for r in recs]
vocab, df = {}, Counter()
for d in docs_t:
    for t in set(d):
        df[t] += 1
for t, c in df.items():
    if c >= 2:
        vocab[t] = len(vocab)
ri, ci, vv = [], [], []
lens = np.zeros(n, dtype=np.float32)
for i, d in enumerate(docs_t):
    tf = Counter(t for t in d if t in vocab)
    lens[i] = sum(tf.values())
    for t, c in tf.items():
        ri.append(i); ci.append(vocab[t]); vv.append(c)
idf = np.array([0.0] * len(vocab), dtype=np.float32)
for t, j in vocab.items():
    idf[j] = np.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
ri = np.array(ri); ci = np.array(ci); tf = np.array(vv, dtype=np.float32)
k1, b, avgl = 1.5, 0.75, lens.mean()
w = idf[ci] * tf * (k1 + 1) / (tf + k1 * (1 - b + b * lens[ri] / avgl))
sp.save_npz("tables_db_bm25.npz", sp.csr_matrix((w, (ri, ci)), shape=(n, len(vocab)), dtype=np.float32))
json.dump(vocab, open(P("tables_db_bm25_vocab.json"), "w", encoding="utf-8"))
print(f"BM25 完成 词表 {len(vocab):,} | 总 {time.time()-t0:.1f}s")
