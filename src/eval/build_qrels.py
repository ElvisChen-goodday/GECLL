"""qrels v3：token 重叠投影（对空白/标点差异鲁棒）+ 相似度分布直方图"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
from config import P, DATA_DIR, ENV_FILE, load_env
import json, re, csv, time
from collections import Counter


WORD = re.compile(r"[a-z][a-z'\-]{3,}")
STOP = set("""the and for that with this from was were are has have had its their they which such
than then when where while also into been being other more most some any all not but can could
may might must shall will would about above after again against among between both during each
few further here how once only own same should there these those through under until very what
when why your our us we you it is of to in on at by as or an be do if""".split())


def nz(s):
    return re.sub(r"\s+", " ", str(s).lower()).strip()


def toks(s):
    return {w for w in WORD.findall(nz(s)) if w not in STOP}


t0 = time.time()
man = list(csv.DictReader(open(P("10k_manifest.csv"), encoding="utf-8-sig")))
tickers = {r["ticker"] for r in man}
CSTOP = {"inc", "corp", "corporation", "company", "co", "plc", "ltd", "limited", "group", "holdings",
         "holding", "the", "and", "of", "trust", "international", "industries", "systems",
         "technologies", "brands", "financial", "energy", "resources", "capital", "partners"}
alias = {}
for r in man:
    words = [w for w in re.sub(r"[^A-Za-z& ]", " ", r["company"]).split()
             if len(w) >= 3 and w.lower() not in CSTOP]
    for k in (3, 2, 1):
        a = " ".join(words[:k]).lower()
        if len(a) >= 5:
            alias.setdefault(a, set()).add(r["ticker"])
alias = {k: list(v)[0] for k, v in alias.items() if len(v) == 1}

parents = {}
with open(P("chunks_struct.jsonl"), encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        if r["level"] == "parent" and r["kind"] == "text":
            parents.setdefault(r["doc_id"], []).append((r["chunk_id"], toks(r["text"])))
print(f"parent {sum(len(v) for v in parents.values()):,} | {time.time()-t0:.0f}s")


def route_doc(text):
    head = " " + nz(text)[:600] + " "
    hits = {t for t in re.findall(r"\b[A-Z]{1,5}\b", str(text)[:300]) if t in tickers}
    best = (0, None)
    for a, t in alias.items():
        if " " + a + " " in head and len(a) > best[0]:
            best = (len(a), t)
    if best[1]:
        hits.add(best[1])
    return list(hits)[0] if len(hits) == 1 else None


rows = [json.loads(l) for l in open(P("references.jsonl"), encoding="utf-8")]
hist = Counter()
out = []
n_ref = n_txt = 0
for r in rows:
    golds = []
    for ref in r["references"]:
        s = str(ref)
        n_ref += 1
        if "|" in s or len(s) < 80 or sum(c.isdigit() for c in s) / max(len(s), 1) >= 0.10:
            continue
        n_txt += 1
        rt = toks(s)
        if len(rt) < 12:
            continue
        doc = route_doc(s)
        if doc is None or doc not in parents:
            hist["route失败"] += 1
            continue
        rt_trunc = rt if len(rt) <= 120 else set(list(rt)[:120])
        best_sc, best_pid = 0.0, None
        for pid, pt in parents[doc]:
            if not pt:
                continue
            sc = len(rt_trunc & pt) / len(rt_trunc)
            if sc > best_sc:
                best_sc, best_pid = sc, pid
        hist[f"{min(int(best_sc*10)/10,0.9):.1f}"] += 1
        if best_sc >= 0.5 and best_pid:
            golds.append({"parent": best_pid, "doc": doc, "score": round(best_sc, 2)})
    if golds:
        out.append({"qid": r["_id"], "query": r["query"], "type": r["type"],
                    "answer": r["answer"], "golds": golds})

json.dump(out, open(P("qrels_text3.json"), "w", encoding="utf-8"), ensure_ascii=False)
print(f"reference {n_ref:,} | 文本型 {n_txt:,} | 可用题 {len(out):,}")
print("最高重叠分数分布：", dict(sorted(hist.items())))
