"""在新 qrels（问句锚定、唯一金标 2,111 题）上量 v4 的 hit@k"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
from config import P, DATA_DIR, ENV_FILE, load_env
import os, re, json, gc, csv, time, unicodedata, argparse
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_OFFLINE", os.environ.get("RAG_OFFLINE", "1"))
import numpy as np
import scipy.sparse as sp
from FlagEmbedding import BGEM3FlagModel, FlagReranker


TOKEN = re.compile(r"[a-z0-9][a-z0-9,\.\-]*")
CSTOP = {"inc", "corp", "corporation", "company", "co", "plc", "ltd", "limited", "group", "holdings",
         "holding", "the", "and", "of", "trust", "international", "industries", "systems",
         "technologies", "brands", "financial", "energy", "resources", "capital", "partners"}


def nz(s):
    return "".join(c for c in unicodedata.normalize("NFKD", str(s))
                   if not unicodedata.combining(c)).lower()


ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=0, help="0 = 全部")
ap.add_argument("--candk", type=int, default=50)
a = ap.parse_args()

man = list(csv.DictReader(open(P("10k_manifest.csv"), encoding="utf-8-sig")))
TICKERS = {r["ticker"] for r in man}
ALIAS = {}
for r in man:
    words = [w for w in re.sub(r"[^A-Za-z& ]", " ", nz(r["company"])).split()
             if len(w) >= 3 and w.lower() not in CSTOP]
    for k in (3, 2, 1):
        s = " ".join(words[:k]).lower()
        if len(s) >= 3:
            ALIAS.setdefault(s, set()).add(r["ticker"])
ALIAS = {k: list(v)[0] for k, v in ALIAS.items() if len(v) == 1}


def route(q):
    cand = {}
    for t in re.findall(r"\b[A-Z]{1,5}\b", q):
        if t in TICKERS:
            cand[t] = 100.0
    ql = " " + re.sub(r"[^a-z0-9& ]", " ", nz(q)) + " "
    qt = set(re.findall(r"[a-z0-9&]+", ql))
    for s, t in ALIAS.items():
        if " " + s + " " in ql or all(x in qt for x in s.split()):
            cand[t] = max(cand.get(t, 0.0), float(len(s)))
    if not cand:
        return None
    best = max(cand.values())
    top = [t for t, v in cand.items() if v == best]
    return top[0] if len(top) == 1 else None


all_ids = [json.loads(l)["chunk_id"] for l in open(P("chunk_ids.jsonl"), encoding="utf-8")]
keep = np.array([i for i, c in enumerate(all_ids) if "-c" in c])
ids = [all_ids[i] for i in keep]
DOC_OF = np.array([c.split(":")[0] for c in ids])
vec = np.load(P("vectors_dense.npy"), mmap_mode="r")
bm25 = sp.load_npz(P("bm25.npz")).tocsr()
VOCAB = json.load(open(P("bm25_vocab.json"), encoding="utf-8"))

qrels = json.load(open(P("qrels_text_v2.json"), encoding="utf-8"))
uniq = [x for x in qrels if all(g["n_hit"] == 1 for g in x["golds"])]
import random
random.seed(11)
qs = uniq if a.n == 0 else random.sample(uniq, min(a.n, len(uniq)))
print(f"评测题 {len(qs):,}（唯一金标池 {len(uniq):,}）", flush=True)

emb = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True, device=os.environ.get("RAG_DEVICE", "cuda"))
cands, golds = [], []
t0 = time.time()
for i, x in enumerate(qs, 1):
    q = x["query"]
    doc = x["doc"]
    subpos = np.where(DOC_OF == doc)[0]
    rows = keep[subpos]
    qv = emb.encode([q], max_length=256)["dense_vecs"][0].astype(np.float32)
    qv /= (np.linalg.norm(qv) + 1e-9)
    ds = np.empty(len(rows), dtype=np.float32)
    for st in range(0, len(rows), 50000):
        blk = rows[st:st + 50000]
        ds[st:st + len(blk)] = np.asarray(vec[blk], dtype=np.float32) @ qv
    toks = [t.strip(".,-").replace(",", "") for t in TOKEN.findall(q.lower())]
    toks = [t for t in toks if len(t) > 1 and t in VOCAB]
    if toks:
        qb = np.zeros(len(VOCAB), dtype=np.float32)
        for t in toks:
            qb[VOCAB[t]] = 1.0
        bs = np.asarray(bm25[rows] @ qb).ravel()
    else:
        bs = np.zeros(len(rows), dtype=np.float32)
    rrf = np.zeros(len(rows), dtype=np.float32)
    for rank, j in enumerate(np.argsort(-ds)[:100]):
        rrf[j] += 1.0 / (60 + rank)
    for rank, j in enumerate(np.argsort(-bs)[:100]):
        rrf[j] += 1.0 / (60 + rank)
    top = np.argsort(-rrf)[:a.candk]
    cands.append([ids[int(subpos[j])] for j in top])
    golds.append(x["golds"][0]["parents"])
    if i % 300 == 0:
        print(f"  召回 {i}/{len(qs)} | {time.time()-t0:.0f}s", flush=True)
print(f"召回完成 {time.time()-t0:.0f}s", flush=True)
del emb
gc.collect()
try:
    import torch
    torch.cuda.empty_cache()
except Exception:
    pass

need = {c for cl in cands for c in cl}
txt = {}
for line in open(P("chunks_struct.jsonl"), encoding="utf-8"):
    if not need:
        break
    r = json.loads(line)
    if r["chunk_id"] in need:
        txt[r["chunk_id"]] = r["text"]
        need.discard(r["chunk_id"])
print(f"候选文本 {len(txt):,}", flush=True)

pool_rank = []
for cl, gs in zip(cands, golds):
    ps = []
    for c in cl:
        pid = c.rsplit("-c", 1)[0]
        if pid not in ps:
            ps.append(pid)
    rk = min((ps.index(g) + 1 for g in gs if g in ps), default=999)
    pool_rank.append(rk)

rer = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True, device=os.environ.get("RAG_DEVICE", "cuda"))
t0 = time.time()
rr_rank = []
for x, cl in zip(qs, cands):
    sc = np.asarray(rer.compute_score([[x["query"], txt.get(c, "")[:1200]] for c in cl],
                                      max_length=512), dtype=np.float32).ravel()
    ps = []
    for i in np.argsort(-sc):
        pid = cl[int(i)].rsplit("-c", 1)[0]
        if pid not in ps:
            ps.append(pid)
    rk = min((ps.index(g) + 1 for g in x["golds"][0]["parents"] if g in ps), default=999)
    rr_rank.append(rk)
print(f"重排完成 {time.time()-t0:.0f}s", flush=True)

n = len(qs)
print(f"\n【parent 级 hit@k】{n} 题（唯一金标）")
print(f"{'k':<6}{'候选池':<10}{'重排后':<10}")
for k in (1, 3, 5, 10, 20, 25, 50):
    print(f"{k:<6}{sum(1 for r in pool_rank if r<=k)/n:<10.0%}{sum(1 for r in rr_rank if r<=k)/n:<10.0%}")
print(f"\nMRR：候选池 {sum((1/r if r<999 else 0) for r in pool_rank)/n:.3f} | "
      f"重排后 {sum((1/r if r<999 else 0) for r in rr_rank)/n:.3f}")
print(f"出池（{a.candk} 个 child 内无金标）：{sum(1 for r in pool_rank if r==999)}/{n} = "
      f"{sum(1 for r in pool_rank if r==999)/n:.0%}")
json.dump({"pool_rank": pool_rank, "rerank_rank": rr_rank},
          open(P("hit_at_k_results.json"), "w"))
