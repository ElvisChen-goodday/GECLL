"""RAG 管线 v4（第 2 步）：rerank 从 parent 改到 child，候选 30 → 50，证据句按 rerank 分选

与 v3 的区别只在重排层：
  - 主路召回 50 个 child（RRF），再用 bge-reranker 逐条打分
  - parent 顺序 = 该 parent 内最高 rerank 分
  - 每个 parent 的证据句 = 该 parent 内 rerank 分最高的 3 个 child
  - 同时记录"只用 RRF 排序"的 parent 顺序，用于对比 P@5
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
from config import P, DATA_DIR, ENV_FILE, load_env, N_PARENTS, CAND_K, SIDE_TOPK, N_EVID
import os, re, json, gc, time, sqlite3, difflib, csv, unicodedata, math
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import scipy.sparse as sp
import requests


YEAR = re.compile(r"\b((?:19|20)\d{2})\b")
NUMONLY = re.compile(r"^[\d\.,\-\(\)\$%\s]+$")
TOKEN = re.compile(r"[a-z0-9][a-z0-9,\.\-]*")
RAW = re.compile(r"-?\(?\$?\d[\d,]*(?:\.\d+)?\)?")
WORD = re.compile(r"[a-zA-Z]{3,}")
SENT = re.compile(r"(?<=[.!?;])\s+")
UNIT = re.compile(r"(?:in|amounts in|dollars in)\s+(thousands|millions|billions)", re.I)
STOP = set("what was were the is are in of for and by on at to total company fiscal year ended"
           " how much many reported report value amount during its their".split())
USE_LOCATE2 = os.environ.get("FR_LOCATE2", "1") == "1"


def nz(s):
    return "".join(c for c in unicodedata.normalize("NFKD", str(s))
                   if not unicodedata.combining(c)).lower()


def nnum(x):
    x = str(x).replace("$", "").replace(",", "").replace(" ", "").strip()
    if x.startswith("(") and x.endswith(")"):
        x = "-" + x[1:-1]
    return x


def same(a, g):
    try:
        return abs(float(nnum(a)) - float(nnum(g))) <= max(0.02, 0.005 * abs(float(nnum(g))))
    except Exception:
        return nnum(a) == nnum(g)


env = load_env()
BASE = env.get("OPENAI_BASE_URL", "https://api.deepseek.com/v1").rstrip("/")
KEY = env.get("OPENAI_API_KEY", "")
MODEL = env.get("OPENAI_MODEL", "deepseek-chat")

man = list(csv.DictReader(open(P("10k_manifest.csv"), encoding="utf-8-sig")))
MAN = {r["ticker"]: r for r in man}
TICKERS = set(MAN)
CSTOP = {"inc", "corp", "corporation", "company", "co", "plc", "ltd", "limited", "group",
         "holdings", "holding", "the", "and", "of", "trust", "international", "industries",
         "systems", "technologies", "brands", "financial", "energy", "resources", "capital",
         "partners"}
ALIAS = {}
for r in man:
    words = [w for w in re.sub(r"[^A-Za-z& ]", " ", nz(r["company"])).split()
             if len(w) >= 3 and w.lower() not in CSTOP]
    for k in (3, 2, 1):
        a = " ".join(words[:k]).lower()
        if len(a) >= 3:
            ALIAS.setdefault(a, set()).add(r["ticker"])
ALIAS = {k: list(v)[0] for k, v in ALIAS.items() if len(v) == 1}


def route(q):
    """显式 ticker(比如括号里的 LNT) 优先级最高，避免被长公司名别名盖过"""
    cand = {}
    for t in re.findall(r"\b[A-Z]{1,5}\b", q):
        if t in TICKERS:
            cand[t] = 100.0
    # 撇号先去掉再切分：Lowe's / Kohl's / Macy's 这类公司名才能和别名表对齐
    qn = nz(q).replace("'", "").replace("’", "").replace("‘", "")
    ql = " " + re.sub(r"[^a-z0-9& ]", " ", qn) + " "
    qtok = set(re.findall(r"[a-z0-9&]+", ql))
    for a, t in ALIAS.items():
        if " " + a + " " in ql or all(x in qtok for x in a.split()):
            cand[t] = max(cand.get(t, 0.0), float(len(a)))
    if not cand:
        return None
    best = max(cand.values())
    top = [t for t, s in cand.items() if s == best]
    return top[0] if len(top) == 1 else None


ids = [json.loads(l)["chunk_id"] for l in open(P("chunk_ids.jsonl"), encoding="utf-8")]
TEXT_MASK = np.array([("-c" in c) for c in ids])
doc_idx = np.load(P("child_doc_idx.npy"))
DOCS = json.load(open(P("child_docs.json")))
DIDX = {d: i for i, d in enumerate(DOCS)}
vec = np.load(P("vectors_dense.npy"), mmap_mode="r")
bm25 = sp.load_npz(P("bm25.npz")).tocsr()
VOCAB = json.load(open(P("bm25_vocab.json"), encoding="utf-8"))

tables = [json.loads(l) for l in open(P("tables_db.jsonl"), encoding="utf-8")]
T_PK = np.array([t["table_pk"] for t in tables])
T_TK = np.array([t["ticker"] for t in tables])
tvec = np.load(P("tables_db_vec.npy"), mmap_mode="r").astype(np.float32)
tbm25 = sp.load_npz(P("tables_db_bm25.npz")).tocsr()
TVOCAB = json.load(open(P("tables_db_bm25_vocab.json"), encoding="utf-8"))
con = sqlite3.connect(P("finder_tables.db"))
cur = con.cursor()

PMETA = {}
for line in open(P("parent_meta.jsonl"), encoding="utf-8"):
    r = json.loads(line)
    PMETA[r["chunk_id"]] = r

T_SEC = {}
for line in open(P("chunks_struct.jsonl"), encoding="utf-8"):
    r = json.loads(line)
    if r["level"] == "parent" and r["kind"] == "table":
        T_SEC.setdefault((r["doc_id"], r["n_rows"]), []).append((r.get("section_path", ""), r["text"][:400]))
print(f"索引就绪 | text child {TEXT_MASK.sum():,} | 表 {len(tables):,}", flush=True)


def qtok(q, vocab):
    out = []
    for x in TOKEN.findall(q.lower()):
        x = x.strip(".,-").replace(",", "")
        if len(x) > 1 and x in vocab:
            out.append(x)
    return out


def main_path(queries):
    from FlagEmbedding import BGEM3FlagModel
    model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True, device="cuda")
    out = []
    for q in queries:
        t0 = time.time()
        doc = route(q)
        sub = np.where(TEXT_MASK & (doc_idx == DIDX[doc]))[0] if doc in DIDX else np.where(TEXT_MASK)[0]
        qv = model.encode([q], max_length=256)["dense_vecs"][0].astype(np.float32)
        qv /= (np.linalg.norm(qv) + 1e-9)
        ds = np.empty(len(sub), dtype=np.float32)
        for st in range(0, len(sub), 50000):
            blk = sub[st:st + 50000]
            ds[st:st + len(blk)] = np.asarray(vec[blk], dtype=np.float32) @ qv
        toks = qtok(q, VOCAB)
        if toks:
            qb = np.zeros(len(VOCAB), dtype=np.float32)
            for t in toks:
                qb[VOCAB[t]] = 1.0
            bs = np.asarray(bm25[sub] @ qb).ravel()
        else:
            bs = np.zeros(len(sub), dtype=np.float32)
        rrf = np.zeros(len(sub), dtype=np.float32)
        for rank, j in enumerate(np.argsort(-ds)[:100]):
            rrf[j] += 1.0 / (60 + rank)
        for rank, j in enumerate(np.argsort(-bs)[:100]):
            rrf[j] += 1.0 / (60 + rank)
        top = np.argsort(-rrf)[:CAND_K]
        out.append({"query": q, "doc": doc, "qv": qv,
                    "cands": [ids[int(sub[j])] for j in top],
                    "rrf_scores": [float(rrf[j]) for j in top]})
    del model
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass
    return out


def rows_of(pk):
    out = {}
    for row, lab, yr, val in cur.execute(
            "select row, row_label, year, value from cells where table_pk=? order by row, col", (pk,)):
        d = out.setdefault(row, {"label": lab or "", "amap": {}})
        if lab and not d["label"]:
            d["label"] = lab.strip()
        if yr and val is not None and yr not in d["amap"]:
            d["amap"][yr] = val
    return out


def locate(pk, q):
    rows = rows_of(pk)
    yrs = YEAR.findall(q)
    year = yrs[-1] if yrs else ""
    ql = " " + re.sub(r"[^a-z0-9& ]", " ", q.lower()) + " "
    cands = []
    for r in sorted(rows):
        d = rows[r]
        lab = d["label"]
        if not lab or NUMONLY.match(lab) or not d["amap"]:
            continue
        ll = " " + re.sub(r"[^a-z0-9& ]", " ", lab.lower()).strip() + " "
        if len(ll.strip()) < 4:
            continue
        if ll in ql:
            sc = 1.0 + min(len(ll), 200) / 1000.0
        else:
            lw = [w for w in ll.split() if len(w) >= 3 and w not in STOP]
            if not lw:
                continue
            hit = sum(1 for w in lw if re.search(r"\b" + re.escape(w), ql))
            sc = 0.9 * hit / len(lw)
            if sc < 0.55:
                sc = max(sc, difflib.SequenceMatcher(None, ll.strip(), ql.strip()).ratio() * 0.7)
        if year in d["amap"]:
            sc += 0.02
        cands.append((sc, len(lab), d, lab))
    if not cands:
        return None, "", 0.0
    cands.sort(key=lambda x: (-x[0], -x[1]))
    sc, _, best, best_lab = cands[0]
    if sc < 0.55 or year not in best["amap"]:
        return None, best_lab, sc
    return best["amap"][year], best_lab, 1.0


# ---------------- locate2：反向覆盖 + IDF 加权，无硬门槛 ----------------
_LTOK = re.compile(r"[a-z][a-z'\-]{2,}")
_LSTOP = set("""the and for with from that this these those was were are has have had its their
they which such than then when where while also into been being other more most some any all
not but can could may might must shall will would about above after again against among between
both during each few further here how once only own same should there through under until very
what why your our us we you it is of to in on at by as or an be do if total company fiscal year
ended much many reported report value amount provided due including include includes""".split())
_LSUF = (("ies", "y"), ("ing", ""), ("ed", ""), ("es", ""), ("s", ""))


def _stem(w):
    for a, b in _LSUF:
        if len(w) > 4 and w.endswith(a):
            return w[: -len(a)] + b
    return w


def _ltok(s):
    return [_stem(t) for t in _LTOK.findall(str(s).lower()) if t not in _LSTOP]


def _build_idf():
    from collections import Counter
    df = Counter()
    n = 0
    for line in open(P("tables_index.jsonl"), encoding="utf-8"):
        r = json.loads(line)
        n += 1
        seen = set()
        for lab in (r.get("labels") or []):
            for t in _ltok(lab):
                if t not in seen:
                    df[t] += 1
                    seen.add(t)
    idf = {t: math.log((n + 1) / (c + 1)) + 1.0 for t, c in df.items()}
    return idf, math.log(n + 1) + 1.0


LIDF, LDEF = _build_idf()


def locate2(pk, q):
    """返回 (value, label, score, year_used)；不设硬门槛，年份只影响选列"""
    rows = rows_of(pk)
    if not rows:
        return None, "", 0.0, None
    yrs = YEAR.findall(q)
    yr = yrs[-1] if yrs else None
    qt = _ltok(q)
    qi = sum(LIDF.get(t, LDEF) for t in qt)
    if not qt or qi <= 0:
        return None, "", 0.0, None
    best = None
    for r in sorted(rows):
        d = rows[r]
        lab = (d["label"] or "").strip()
        amap = d.get("amap") or {}
        if not lab or not amap:
            continue
        lt = _ltok(lab)
        if not lt:
            continue
        ls = set(lt)
        cov = sum(LIDF.get(t, LDEF) for t in qt if t in ls) / qi
        lab_cov = len(ls & set(qt)) / len(ls)
        sc = 0.75 * cov + 0.25 * lab_cov
        if yr and yr in amap:
            sc += 0.03
        if best is None or sc > best[0]:
            best = (sc, lab, amap)
    if best is None or best[0] <= 0:
        return None, "", 0.0, None
    sc, lab, amap = best
    use_year = yr if (yr and yr in amap) else (sorted(amap)[-1] if amap else None)
    return amap.get(use_year), lab, round(sc, 3), use_year


def fact_record(pk, row_label, value):
    doc_id, n_rows, caption = cur.execute(
        "select doc_id, n_rows, caption from tables where table_pk=?", (pk,)).fetchone() or (None, None, "")
    cells = cur.execute("""select row, row_label, year, text, value from cells
                           where table_pk=? and value is not null order by row, col""", (pk,)).fetchall()
    hits = [c for c in cells if c[4] is not None and abs(c[4] - float(nnum(value))) < 1e-6]
    row = hits[0][0] if hits else None
    year = next((c[2] for c in hits if c[2]), None)
    raw = hits[0][3] if hits else None
    rowctx = {}
    if row is not None:
        for c in cells:
            if c[0] == row and c[2] and c[4] is not None:
                rowctx.setdefault(c[2], c[4])
    m = UNIT.search(caption or "")
    if not m:
        hdr = " ".join(str(x[0]) for x in cur.execute(
            "select text from cells where table_pk=? and row<3 limit 40", (pk,)).fetchall() if x[0])
        m = UNIT.search(hdr)
    unit = f"USD {m.group(1).lower()}" if m else "unknown"
    kind = "percent" if (raw and "%" in str(raw)) else ("currency" if unit.startswith("USD") else "number")
    if "share" in (row_label or "").lower():
        kind = "shares"
    sec = None
    for s, txt in T_SEC.get((doc_id, n_rows), []):
        if row_label and row_label.lower()[:40] in txt.lower():
            sec = s
            break
    if not sec:
        cand = T_SEC.get((doc_id, n_rows), [])
        sec = cand[0][0] if len(cand) == 1 else None
    m2 = MAN.get(doc_id, {})
    return {"metric": row_label, "value": value, "raw": raw, "year": year,
            "year_candidates": sorted(rowctx), "row_context": rowctx, "unit": unit, "kind": kind,
            "section": sec, "ticker": doc_id, "fiscal_year": m2.get("fiscal_year"),
            "period_end": m2.get("period_end")}


def load_texts(need):
    need = set(need)
    got = {}
    with open(P("chunks_struct.jsonl"), encoding="utf-8") as f:
        for line in f:
            if not need:
                break
            r = json.loads(line)
            if r["chunk_id"] in need:
                got[r["chunk_id"]] = r["text"]
                need.discard(r["chunk_id"])
    return got


def rerank_all(items, ctext):
    """一次加载 reranker，同时完成：child 重排（主路）+ 表指纹重排（旁路）"""
    from FlagEmbedding import FlagReranker
    rer = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True, device="cuda")
    for it in items:
        q = it["query"]
        t0 = time.time()
        pairs = [[q, ctext.get(c, "")[:1200]] for c in it["cands"]]
        sc = np.asarray(rer.compute_score(pairs, max_length=512), dtype=np.float32).ravel()
        order = np.argsort(-sc)

        evid, rrf_parents = {}, []
        for i in order:
            cid = it["cands"][int(i)]
            pid = cid.rsplit("-c", 1)[0]
            if pid not in evid:
                evid[pid] = {"children": [], "score": float(sc[int(i)])}
                rrf_parents.append(pid)
            if len(evid[pid]["children"]) < N_EVID:
                evid[pid]["children"].append(cid)
        it["parents"] = rrf_parents[:N_PARENTS]
        it["evid"] = evid
        # 只用 RRF 顺序的 parent，做对比
        rp, seen = [], set()
        for c in it["cands"]:
            pid = c.rsplit("-c", 1)[0]
            if pid in seen:
                continue
            seen.add(pid)
            rp.append(pid)
        it["parents_rrf"] = rp[:N_PARENTS]
        it["rerank_ms"] = round((time.time() - t0) * 1000)

        # 旁路
        doc, qv = it["doc"], it["qv"]
        if doc is None or doc not in set(T_TK):
            it["facts"] = []
            continue
        idxs = np.where(T_TK == doc)[0]
        ds = tvec[idxs] @ qv
        toks = qtok(q, TVOCAB)
        if toks:
            qb = np.zeros(len(TVOCAB), dtype=np.float32)
            for t in toks:
                qb[TVOCAB[t]] = 1.0
            bs = np.asarray(tbm25[idxs] @ qb).ravel()
        else:
            bs = np.zeros(len(idxs), dtype=np.float32)
        rrf2 = np.zeros(len(idxs), dtype=np.float32)
        for rank, j in enumerate(np.argsort(-ds)[:100]):
            rrf2[j] += 1.0 / (60 + rank)
        for rank, j in enumerate(np.argsort(-bs)[:100]):
            rrf2[j] += 1.0 / (60 + rank)
        top = idxs[np.argsort(-rrf2)[:30]]
        sc2 = np.asarray(rer.compute_score([[q, tables[j]["fingerprint"]] for j in top],
                                           max_length=512), dtype=np.float32).ravel()
        votes = {}
        for j in top[np.argsort(-sc2)[:SIDE_TOPK]]:
            pk = int(T_PK[j])
            if USE_LOCATE2:
                val, lab, conf, _yr = locate2(pk, q)
            else:
                val, lab, conf = locate(pk, q)
            if val is None:
                continue
            k = round(float(nnum(val)), 4)
            v = votes.setdefault(k, {"value": val, "label": lab, "pk": pk, "n": 0})
            v["n"] += 1
        facts = []
        for k, v in sorted(votes.items(), key=lambda kv: -kv[1]["n"])[:2]:
            rec = fact_record(v["pk"], v["label"], v["value"])
            rec["votes"] = v["n"]
            facts.append(rec)
        it["facts"] = facts
    del rer
    gc.collect()
    return items


def pick_evidence(child_text, query, k=2, cap=520):
    sents = [s.strip() for s in SENT.split(child_text) if len(s.strip()) > 20]
    if not sents:
        return child_text[:cap]
    qw = {w.lower() for w in WORD.findall(query) if w.lower() not in STOP}
    scored = []
    for s in sents:
        sw = {w.lower() for w in WORD.findall(s)}
        sc = len(qw & sw) / max(len(qw), 1)
        if re.search(r"\d", s):
            sc += 0.15
        scored.append((sc, s))
    scored.sort(key=lambda x: -x[0])
    out, tot = [], 0
    for sc, s in scored[:k]:
        if tot + len(s) > cap:
            break
        out.append(s)
        tot += len(s)
    return " ".join(out) if out else sents[0][:cap]


def build_ctx(it, texts):
    parts = []
    if it["facts"]:
        parts.append("【取数结果（表格旁路，字段已对齐）】")
        for f in it["facts"]:
            parts.append(f"- metric: {f['metric']} | value: {f['value']} | raw: {f['raw']} | year: {f['year']} "
                         f"| year_candidates: {f['year_candidates']} | unit: {f['unit']} | kind: {f['kind']} "
                         f"| votes: {f['votes']}/5 | ticker: {f['ticker']} | fiscal_year: {f['fiscal_year']} "
                         f"| period_end: {f['period_end']}\n  section: {str(f['section'])[:110]}"
                         f"\n  row_context: {f['row_context']}")
    parts.append("【章节骨架 + 证据句（按相关性模型打分排序）】")
    for n, pid in enumerate(it["parents"], 1):
        m = PMETA.get(pid, {})
        full = texts.get(pid, "")
        sents = [s.strip() for s in SENT.split(full) if s.strip()]
        block = (f"[{n}] {pid} | role={m.get('role')} | ticker={m.get('doc_id')} | "
                 f"fiscal_year={MAN.get(m.get('doc_id'), {}).get('fiscal_year')} | "
                 f"period_end={MAN.get(m.get('doc_id'), {}).get('period_end')} | years={m.get('years')}\n"
                 f"  section: {m.get('section_path', '')[:150]}\n"
                 f"  主题句: {sents[0][:200] if sents else ''}")
        for i, cid in enumerate(it["evid"].get(pid, {}).get("children", []), 1):
            ct = texts.get(cid, "")
            if not ct:
                continue
            sc = it["evid"][pid].get("score", 0)
            block += f"\n  ▶ 证据{i} [{cid} r={sc:.2f}]: {pick_evidence(ct, it['query'])}"
        block += f"\n  （本段共 {len(sents)} 句，此处为摘录；prev={m.get('prev_id')} next={m.get('next_id')}）"
        parts.append(block)
    return "\n\n".join(parts)


def answer(ctx, q):
    prompt = (
        "你是财务分析助手。材料由「取数结果」与「章节骨架 + 证据句」组成；证据句已按相关性模型排序。\n"
        "规则：\n"
        "1) 取数结果有值就直接用，写明 year 与 unit；unit=unknown 写“单位未在表中声明”。\n"
        "2) year_candidates 与问题年份不符 → 指出歧义。\n"
        "3) 证据句能找到就引用，标 [段号]；role 只是参考线索，不作为排除依据。\n"
        "4) 证据不足但主题相关 → 给最接近事实并说明可能不完整，不要直接拒答。\n"
        "5) 只有主题完全无关才 answerable=false。\n"
        "输出 JSON：{\"answerable\": true/false, \"reason\": \"fact|evidence|partial|no_evidence\", "
        "\"answer\": \"...\", \"evidence\": [\"[1] 关键句\"], \"used_fields\": [...]}\n\n"
        + ctx[:14000] + "\n\n问题：" + q)
    r = requests.post(BASE + "/chat/completions", timeout=120,
                      headers={"Authorization": "Bearer " + KEY},
                      json={"model": MODEL, "temperature": 0,
                            "messages": [{"role": "user", "content": prompt}]})
    r.raise_for_status()
    out = r.json()["choices"][0]["message"]["content"]
    m = re.search(r"\{.*\}", out, re.S)
    try:
        return json.loads(m.group(0)) if m else {"answerable": None, "answer": out}
    except Exception:
        return {"answerable": None, "answer": out}



# ==================== 对外入口 ====================
def ask(question, want_debug=False):
    """单次问答：route → 主路(召回+重排) → 旁路(取数) → 组装 → LLM"""
    import tempfile
    items = main_path([question])
    items[0]["kind"] = "text"
    ctext = load_texts({c for it in items for c in it["cands"]})
    items = rerank_all(items, ctext)
    ptext = load_texts({p for it in items for p in it["parents"]})
    ctx = build_ctx(items[0], {**ctext, **ptext})
    obj = answer(ctx, question)
    out = {"query": question, "ticker": items[0]["doc"], **obj}
    if want_debug:
        out["debug"] = {"parents": items[0]["parents"], "facts": items[0]["facts"],
                        "ctx_chars": len(ctx), "cands": items[0]["cands"][:10]}
    return out


def _eval20():
    import random
    qrels = json.load(open(P("qrels_text3.json"), encoding="utf-8"))
    tq = json.load(open("no_llm_cands.json", encoding="utf-8"))
    random.seed(77)
    t_qs = [{"query": x["query"], "gold_parent": x["golds"][0]["parent"], "gold_answer": x["answer"],
             "kind": "text"} for x in random.sample(qrels, 10)]
    random.seed(78)
    b_qs = [{"query": x["query"], "gold_value": x["gold"], "gold_answer": str(x["gold"]),
             "kind": "table"} for x in random.sample(tq, 10)]
    items_q = t_qs + b_qs

    items = main_path([x["query"] for x in items_q])
    for it, meta in zip(items, items_q):
        it.update({k: v for k, v in meta.items() if k != "query"})
    ctext = load_texts({c for it in items for c in it["cands"]})
    items = rerank_all(items, ctext)
    ptext = load_texts({p for it in items for p in it["parents"]})
    rows = []
    for it in items:
        ctx = build_ctx(it, {**ctext, **ptext})
        it["answer_obj"] = answer(ctx, it["query"])
        it["ctx_chars"] = len(ctx)
        rows.append(it)
    json.dump([{k: v for k, v in it.items() if k != "qv"} for it in rows],
              open(P("finder_rag_eval20.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    p5 = sum(1 for it in rows if it["kind"] == "text" and it["gold_parent"] in it["parents"])
    refuse = sum(1 for it in rows if it["answer_obj"].get("answerable") is False)
    print(f"拒答 {refuse}/20 | 文本 P@5 {p5}/10 | 平均上下文 {sum(i['ctx_chars'] for i in rows)/20:,.0f}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--eval":
        _eval20()
    elif len(sys.argv) > 1 and sys.argv[1] == "--ask":
        q = " ".join(sys.argv[2:])
        r = ask(q, want_debug=True)
        print(json.dumps(r, ensure_ascii=False, indent=1))
    else:
        print("用法: python -m src.rag.retriever --ask \"What was Lowe's cash dividend payments in 2023?\"")
        print("      python -m src.rag.retriever --eval")
