"""为 551,692 个 child 建 BM25 索引（scipy 稀疏矩阵，避免 rank_bm25 的 dict 爆炸）

输出：
  bm25.npz    csr_matrix(float32)  + 词表
  bm25_vocab.json
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
from config import P, DATA_DIR, ENV_FILE, load_env
import json, re, time
from collections import Counter
import numpy as np
import scipy.sparse as sp


TOKEN = re.compile(r"[a-z0-9][a-z0-9,\.\-]*")


def tok(text):
    out = []
    for t in TOKEN.findall(text.lower()):
        t = t.strip(".,-")
        if len(t) > 1:
            out.append(t.replace(",", ""))
    return out


t0 = time.time()
ids, docs = [], []
with open(P("chunks_struct.jsonl"), encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        if r["level"] == "child":
            ids.append(r["chunk_id"])
            docs.append(tok(r["text"]))
n = len(docs)
print(f"文档 {n:,} | 分词耗时 {time.time()-t0:.1f}s", flush=True)

vocab = {}
df = Counter()
for d in docs:
    for t in set(d):
        df[t] += 1
for t in df:
    if df[t] >= 2:                      # 只保留出现 ≥2 次的词，压词表
        vocab[t] = len(vocab)
print(f"词表 {len(vocab):,}（df≥2）| 累计 {time.time()-t0:.1f}s", flush=True)

rows_i, cols_i, vals = [], [], []
lens = np.zeros(n, dtype=np.float32)
k1, b = 1.5, 0.75
for i, d in enumerate(docs):
    tf = Counter(t for t in d if t in vocab)
    lens[i] = sum(tf.values())
    for t, c in tf.items():
        rows_i.append(i)
        cols_i.append(vocab[t])
        vals.append(c)
print(f"非零项 {len(vals):,} | 累计 {time.time()-t0:.1f}s", flush=True)

tf_mat = sp.csr_matrix((np.array(vals, dtype=np.float32), (rows_i, cols_i)),
                       shape=(n, len(vocab)))
avg = lens.mean()
idf = np.zeros(len(vocab), dtype=np.float32)
for t, j in vocab.items():
    dft = df[t]
    idf[j] = np.log(1 + (n - dft + 0.5) / (dft + 0.5))

# BM25 权重：逐元素作用在 tf 上
rows_i = np.array(rows_i)
cols_i = np.array(cols_i)
tf = np.array(vals, dtype=np.float32)
denom = tf + k1 * (1 - b + b * lens[rows_i] / avg)
w = idf[cols_i] * tf * (k1 + 1) / denom
bm25 = sp.csr_matrix((w, (rows_i, cols_i)), shape=(n, len(vocab)), dtype=np.float32)
print(f"BM25 矩阵 {bm25.shape} | nnz {bm25.nnz:,} | {time.time()-t0:.1f}s", flush=True)

sp.save_npz("bm25.npz", bm25)
json.dump(vocab, open(P("bm25_vocab.json"), "w", encoding="utf-8"))
json.dump({"n": n, "avgdl": float(avg)}, open("bm25_meta.json", "w"))
print(f"保存完成 | 总耗时 {time.time()-t0:.1f}s | 约 {bm25.data.nbytes/1e6:.0f} MB 权重")
