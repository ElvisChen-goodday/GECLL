"""707 题：dense+BM25+rerank 选表 → 渲染 top-2 表（修 caption / 单位 / 行选择）

输出 ctx_707_dense_render.jsonl
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from config import P, DATA_DIR, ENV_FILE, load_env
import os, sys, json, re, time, gc
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np
import FlagEmbedding


def make_cached(cls):
    cache = {}

    def f(*a, **kw):
        key = (a[0] if a else kw.get("model_name_or_path"), str(sorted(kw.items())))
        if key not in cache:
            print(f"  [model] 加载 {key[0]}", flush=True)
            cache[key] = cls(*a, **kw)
        return cache[key]

    return f


FlagEmbedding.BGEM3FlagModel = make_cached(FlagEmbedding.BGEM3FlagModel)
FlagEmbedding.FlagReranker = make_cached(FlagEmbedding.FlagReranker)

from src.rag import retriever as fr  # noqa: E402


OUT = P("ctx_707_dense_render.jsonl")
LIM = 707
for i, a in enumerate(sys.argv):
    if a == "--limit":
        LIM = int(sys.argv[i + 1])

UNIT = re.compile(r"(?:in|amounts in|dollars in)\s+(thousands|millions|billions)", re.I)
JUNK = re.compile(r"^[_\-—=*·.\s]+$")


def meta_of(pk):
    return fr.cur.execute(
        "select doc_id, caption, section, n_rows, n_cols from tables where table_pk=?",
        (pk,)).fetchone() or (None, "", "", 0, 0)


def clean_cap(cap):
    cap = (cap or "").strip()
    if not cap or JUNK.match(cap):
        return ""
    parts = [p.strip() for p in re.split(r"\s{2,}|\n", cap) if p and not JUNK.match(p)]
    return parts[0][:100] if parts else ""


def unit_of(pk):
    doc_id, cap, sec, nr, nc = meta_of(pk)
    m = UNIT.search(cap or "")
    if not m:
        hdr = " ".join(str(x[0]) for x in fr.cur.execute(
            "select text from cells where table_pk=? and row<3 limit 40", (pk,)).fetchall() if x[0])
        m = UNIT.search(hdr)
    return (m.group(1).lower() if m else ""), clean_cap(cap), sec or "", doc_id


def render(pk, q, max_rows=26):
    rows = fr.rows_of(pk)
    if not rows:
        return ""
    items = []
    for r in sorted(rows):
        d = rows[r]
        lab = (d.get("label") or "").strip()
        amap = d.get("amap") or {}
        if not lab or not amap:
            continue
        items.append((lab, amap))
    if not items:
        return ""
    years = sorted({y for _, amap in items for y in amap})[-4:]
    u, cap, sec, doc_id = unit_of(pk)
    qt = set(fr._ltok(q))

    def rel(lab):
        ls = set(fr._ltok(lab))
        return len(ls & qt) / max(len(ls), 1)

    order = sorted(range(len(items)), key=lambda i: -rel(items[i][0]))
    keep = [i for i in order[:20]]
    keep = sorted(keep)[:max_rows]
    head = f"表：{doc_id}｜{cap or '(无标题)'}｜单位：{('USD ' + u) if u else '未声明'}"
    if sec:
        head += f"｜章节：{sec[:50]}"
    lines = [head, "行标签 | " + " | ".join(years)]
    for i in keep:
        lab, amap = items[i]
        cells = [str(amap.get(y)) if amap.get(y) is not None else "-" for y in years]
        lines.append(f"{lab[:70]} | " + " | ".join(cells))
    return "\n".join(lines)[:2600]


old = {json.loads(l)["qid"]: json.loads(l) for l in open(P("ctx_all.jsonl"), encoding="utf-8")}
tbl = [c for c in old.values() if c.get("tbl")]
done = set()
if os.path.exists(OUT):
    for line in open(OUT, encoding="utf-8"):
        try:
            done.add(json.loads(line)["qid"])
        except Exception:
            pass
todo = [c for c in tbl if c["qid"] not in done][:LIM]
print(f"表特征题 {len(tbl)} | 已完成 {len(done)} | 本轮 {len(todo)}", flush=True)
if not todo:
    print("ALLDONE", flush=True)
    raise SystemExit(0)

T_TK, T_PK, tvec, tbm25, TVOCAB = fr.T_TK, fr.T_PK, fr.tvec, fr.tbm25, fr.TVOCAB
from FlagEmbedding import BGEM3FlagModel, FlagReranker  # noqa: E402

enc = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True, device="cuda")
rer = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True, device="cuda")

B = 60
f = open(OUT, "a", encoding="utf-8")
t0 = time.time()
n_tbl = 0
for s in range(0, len(todo), B):
    batch = todo[s:s + B]
    qs = [c["query"] for c in batch]
    qvs = enc.encode(qs, max_length=256, batch_size=16)["dense_vecs"].astype(np.float32)
    qvs /= (np.linalg.norm(qvs, axis=1, keepdims=True) + 1e-9)
    for c, qv in zip(batch, qvs):
        doc, q = c.get("doc"), c["query"]
        blocks = []
        if doc and doc in set(T_TK):
            idxs = np.where(T_TK == doc)[0]
            ds = tvec[idxs] @ qv
            toks = fr.qtok(q, TVOCAB)
            if toks:
                qb = np.zeros(len(TVOCAB), dtype=np.float32)
                for t in toks:
                    qb[TVOCAB[t]] = 1.0
                bs = np.asarray(tbm25[idxs] @ qb).ravel()
            else:
                bs = np.zeros(len(idxs), dtype=np.float32)
            rrf = np.zeros(len(idxs), dtype=np.float32)
            for rank, j in enumerate(np.argsort(-ds)[:100]):
                rrf[j] += 1.0 / (60 + rank)
            for rank, j in enumerate(np.argsort(-bs)[:100]):
                rrf[j] += 1.0 / (60 + rank)
            top = idxs[np.argsort(-rrf)[:30]]
            if len(top):
                sc = rer.compute_score([[q, fr.tables[j]["fingerprint"]] for j in top], max_length=512)
                sc = np.asarray(sc, dtype=np.float32).ravel()
                for j in top[np.argsort(-sc)[:2]]:
                    txt = render(int(T_PK[j]), q)
                    if txt:
                        blocks.append(txt)
        n_tbl += 1 if blocks else 0
        body = c["ctx"]
        if "【章节骨架" in body:
            body = body[body.index("【章节骨架"):]
        if blocks:
            body = "【命中表格（dense+rerank 选表，按年份列渲染）】\n" + "\n\n".join(blocks) + "\n\n" + body
        f.write(json.dumps({"qid": c["qid"], "query": q, "gold": c["gold"],
                            "category": c.get("category"), "doc": doc, "tbl": True,
                            "ctx": body, "facts_n": len(blocks)}, ensure_ascii=False) + "\n")
    print(f"  {min(s+B, len(todo))}/{len(todo)} | {time.time()-t0:.0f}s | 有表 {n_tbl}", flush=True)
    gc.collect()
f.close()
print(f"本轮完成 {time.time()-t0:.0f}s | 有表 {n_tbl}/{len(todo)} = {n_tbl/max(len(todo),1):.0%}", flush=True)
