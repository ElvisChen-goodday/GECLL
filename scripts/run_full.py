"""全库跑（4,813 非计算题）：分三段、全可续跑

Phase 1 (GPU, 串行)  ctx_all.jsonl     检索+重排+上下文，每 100 题落盘
Phase 2 (线程 8)     answers_full.jsonl  v9 prompt 作答
Phase 3 (线程 8)     verdicts_full.jsonl 判分
Phase 4             汇总报告

用法：
  python run_full.py --phase ctx      # 只建上下文
  python run_full.py --phase gen      # 只作答（可重复跑，自动跳过已完成）
  python run_full.py --phase judge
  python run_full.py --phase report
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from config import P, DATA_DIR, ENV_FILE, load_env
from src.generate.prompts import ANSWER_SYS as SYS, JUDGE_SYS as JUDGE, REPAIR_SYS
import os, re, json, time, random, sys, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import requests


PHASE = None
for i, a in enumerate(sys.argv):
    if a == "--phase":
        PHASE = sys.argv[i + 1]
PHASE = PHASE or "all"

CTX_F = P("ctx_all.jsonl")
ANS_F = P("answers_full.jsonl")
VER_F = P("verdicts_final.jsonl") if __import__("os").path.exists(P("verdicts_final.jsonl")) else P("verdicts_full.jsonl")
LOCK = threading.Lock()


def load_jsonl(p):
    if not os.path.exists(p):
        return {}
    out = {}
    for line in open(p, encoding="utf-8"):
        try:
            r = json.loads(line)
        except Exception:
            continue
        out[r.get("qid") or r.get("_id")] = r
    return out


def append(p, rec):
    with LOCK:
        with open(p, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def table_like(r):
    refs = " ".join(str(x) for x in r["references"])
    if "|" in refs:
        return True
    return sum(c.isdigit() for c in refs) / max(len(refs), 1) > 0.10


rows = [json.loads(l) for l in open(P("references.jsonl"), encoding="utf-8")]
pool = [r for r in rows if str(r.get("type")) == "None" and (r.get("answer") or "").strip()]
print(f"全库 {len(rows)} | 非计算且有金标 {len(pool)}", flush=True)

# ---------- Phase 1：上下文 ----------
if PHASE in ("all", "ctx"):
    from src.rag import retriever as fr
    import gc
    try:
        import torch
    except Exception:
        torch = None
    done = load_jsonl(CTX_F)
    todo = [r for r in pool if r["_id"] not in done]
    print(f"[ctx] 已完成 {len(done)} | 待建 {len(todo)}", flush=True)
    B = 30
    t0 = time.time()
    for s in range(0, len(todo), B):
        batch = todo[s:s + B]
        try:
            items = fr.main_path([x["query"] for x in batch])
            ctext = fr.load_texts({c for it in items for c in it["cands"]})
            items = fr.rerank_all(items, ctext)
            ptext = fr.load_texts({p for it in items for p in it["parents"]})
            for it, meta in zip(items, batch):
                ctx = fr.build_ctx(it, {**ctext, **ptext})
                append(CTX_F, {"qid": meta["_id"], "query": it["query"], "gold": meta["answer"],
                               "category": meta.get("category"), "doc": it["doc"],
                               "parents": it["parents"], "ctx": ctx,
                               "tbl": table_like(meta)})
            del items, ctext, ptext
        except Exception as e:
            print(f"  批次 {s} 异常（跳过）: {str(e)[:160]}", flush=True)
        gc.collect()
        if torch is not None:
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        el = time.time() - t0
        done_n = len(done) + min(s + B, len(todo))
        rate = el / max(s + B, 1)
        print(f"  [ctx] {done_n}/{len(pool)} | {el:.0f}s | 剩余 {rate*(len(todo)-s-B)/60:.0f} 分钟",
              flush=True)

# ---------- Phase 2：作答 ----------
BASE = KEY = MODEL = None


def init_llm():
    global BASE, KEY, MODEL
    from src.rag import retriever as fr
    BASE, KEY, MODEL = fr.BASE, fr.KEY, fr.MODEL





def call(prompt, timeout=180):
    r = requests.post(BASE + "/chat/completions", timeout=timeout,
                      headers={"Authorization": "Bearer " + KEY},
                      json={"model": MODEL, "temperature": 0,
                            "messages": [{"role": "user", "content": prompt}]})
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def parse(txt):
    m = re.search(r"\{.*\}", txt, re.S)
    if not m:
        return {"answerable": None, "answer": txt}
    try:
        return json.loads(m.group(0))
    except Exception:
        return {"answerable": None, "answer": txt}


if PHASE in ("all", "gen"):
    init_llm()
    ctxs = load_jsonl(CTX_F)
    done = load_jsonl(ANS_F)
    todo = [c for q, c in ctxs.items() if q not in done]
    print(f"[gen] 上下文 {len(ctxs)} | 已完成 {len(done)} | 待作答 {len(todo)}", flush=True)
    t0 = time.time()
    cnt = [0]

    def work(c):
        obj = parse(call(SYS + "\n\n" + c["ctx"][:14000] + "\n\n问题：" + c["query"]))
        usable = [f for f in (obj.get("facts") or []) if f.get("usable") is not False]
        if obj.get("answerable") is False and usable:
            o2 = parse(call("你上一条输出自相矛盾：存在 usable 的 facts 却 answerable=false。"
                            "按规则应改判 true 并作答。只输出修正后的 JSON。\n\n原问题：" + c["query"] +
                            "\nfacts：" + json.dumps(obj.get("facts"), ensure_ascii=False)[:1200]))
            if o2.get("answerable") is True:
                o2.setdefault("facts", obj.get("facts"))
                obj = o2
        rec = {"qid": c["qid"], "query": c["query"], "gold": c["gold"], "category": c["category"],
               "tbl": c.get("tbl", False), "answer_obj": obj}
        append(ANS_F, rec)
        with LOCK:
            cnt[0] += 1
            if cnt[0] % 100 == 0:
                el = time.time() - t0
                print(f"  [gen] {cnt[0]}/{len(todo)} | {el:.0f}s | "
                      f"剩余 {el/cnt[0]*(len(todo)-cnt[0])/60:.0f} 分钟", flush=True)
        return True

    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(work, c) for c in todo]
        for f in as_completed(futs):
            try:
                f.result()
            except Exception as e:
                print("  gen 异常:", str(e)[:120], flush=True)
    print(f"[gen] 完成 {len(load_jsonl(ANS_F))}", flush=True)

# ---------- Phase 3：判分 ----------



def judge_one(rec):
    ans = str(rec["answer_obj"].get("answer") or "")
    try:
        txt = call(JUDGE + "问题：" + str(rec["query"]) + "\n金标答案：" + str(rec["gold"])[:1500] +
                   "\n模型答案：" + ans[:2000], timeout=90)
    except Exception:
        return "error"
    m = re.search(r"\{.*\}", txt, re.S)
    try:
        return json.loads(m.group(0)).get("verdict", "?")
    except Exception:
        return "?"


if PHASE in ("all", "judge"):
    init_llm()
    ans = load_jsonl(ANS_F)
    done = load_jsonl(VER_F)
    todo = [a for q, a in ans.items() if q not in done]
    print(f"[judge] 答案 {len(ans)} | 已判 {len(done)} | 待判 {len(todo)}", flush=True)
    t0 = time.time()

    def wj(a):
        v = judge_one(a)
        append(VER_F, {"qid": a["qid"], "verdict": v, "category": a.get("category"),
                       "tbl": a.get("tbl", False), "query": a["query"], "gold": a["gold"],
                       "answer": str(a["answer_obj"].get("answer") or "")})
        return v

    n = [0]
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(wj, a) for a in todo]
        for f in as_completed(futs):
            try:
                f.result()
                n[0] += 1
                if n[0] % 200 == 0:
                    print(f"  [judge] {n[0]}/{len(todo)} | {time.time()-t0:.0f}s", flush=True)
            except Exception as e:
                print("  judge 异常:", str(e)[:120], flush=True)
    print(f"[judge] 完成 {len(load_jsonl(VER_F))}", flush=True)

# ---------- Phase 4：报告 ----------
if PHASE in ("all", "report"):
    from collections import Counter, defaultdict
    ver = load_jsonl(VER_F)
    n = len(ver)
    if not n:
        print("无判分结果")
        sys.exit(0)
    for grp, f in (("全部非计算题", 1.0), ("文本题", False), ("表特征题", True)):
        if grp == "全部非计算题":
            sub = list(ver.values())
        else:
            sub = [r for r in ver.values() if bool(r.get("tbl")) == f]
        if not sub:
            continue
        c = Counter(r["verdict"] for r in sub)
        m = len(sub)
        print(f"\n【{grp}】n={m}")
        print(f"  correct  {c['correct']:>5}/{m} = {c['correct']/m:.1%}")
        print(f"  +partial {c['correct']+c['partial']:>5}/{m} = {(c['correct']+c['partial'])/m:.1%}")
        print(f"  wrong {c['wrong']:>4} ({c['wrong']/m:.1%}) | refused {c['refused']:>4} "
              f"({c['refused']/m:.1%}) | 其他 {c['?']+c['error']}")
    bycat = defaultdict(Counter)
    for r in ver.values():
        bycat[r.get("category") or "?"][r["verdict"]] += 1
    print("\n分类别（correct | +partial）：")
    for k, c in sorted(bycat.items(), key=lambda kv: -sum(kv[1].values())):
        m = sum(c.values())
        print(f"  {k:<24}{c['correct']/m:>6.0%} | {(c['correct']+c['partial'])/m:>6.0%}   (n={m})")
