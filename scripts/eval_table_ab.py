"""707 题旁路 A/B：old vs new 行匹配，各跑一轮作答 + 判分（不需要模型/GPU）

输入：ctx_707_bm25_old.jsonl / ctx_707_bm25_new.jsonl
输出：answers_707_<mode>.jsonl, verdicts_707_<mode>.jsonl
对照：verdicts_full.jsonl 中同一批 qid 的现状结果
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from config import P, DATA_DIR, ENV_FILE, load_env
from src.generate.prompts import ANSWER_SYS as SYS, JUDGE_SYS as JUDGE, REPAIR_SYS
import os, re, json, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

MODES = ["old", "new"]
for i, a in enumerate(sys.argv):
    if a == "--mode":
        MODES = [sys.argv[i + 1]]

BASE = KEY = MODEL = None
from src.rag import retriever as fr  # 只取 BASE/KEY/MODEL

BASE, KEY, MODEL = fr.BASE, fr.KEY, fr.MODEL
import requests


LOCK = threading.Lock()






def call(prompt, timeout=180, tries=4):
    last = None
    for k in range(tries):
        try:
            r = requests.post(BASE + "/chat/completions", timeout=timeout,
                              headers={"Authorization": "Bearer " + KEY},
                              json={"model": MODEL, "temperature": 0,
                                    "messages": [{"role": "user", "content": prompt}]})
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        except Exception as e:
            last = e
            time.sleep(2 + 3 * k)
    raise last


def parse(txt):
    m = re.search(r"\{.*\}", txt, re.S)
    if not m:
        return {"answerable": None, "answer": txt}
    try:
        return json.loads(m.group(0))
    except Exception:
        return {"answerable": None, "answer": txt}


def judge(q, gold, ans):
    try:
        t = call(JUDGE + "问题：" + str(q) + "\n金标答案：" + str(gold)[:1500] + "\n模型答案：" + str(ans)[:2000],
                 timeout=90)
    except Exception:
        return "error"
    m = re.search(r"\{.*\}", t, re.S)
    try:
        return json.loads(m.group(0)).get("verdict", "?")
    except Exception:
        return "?"


base = {}
if os.path.exists(P("verdicts_full.jsonl")):
    for line in open(P("verdicts_full.jsonl"), encoding="utf-8"):
        r = json.loads(line)
        base[r["qid"]] = r["verdict"]

from collections import Counter
for mode in MODES:
    src = P(f"ctx_707_bm25_{mode}.jsonl")
    if not os.path.exists(src):
        print(f"缺少 {src}")
        continue
    recs = [json.loads(l) for l in open(src, encoding="utf-8")]
    ap = P(f"answers_707_{mode}.jsonl")
    done_q = set()
    if os.path.exists(ap):
        for line in open(ap, encoding="utf-8"):
            try:
                done_q.add(json.loads(line)["qid"])
            except Exception:
                pass
    todo_recs = [r for r in recs if r["qid"] not in done_q]
    print(f"\n=== mode={mode} | 待作答 {len(todo_recs)}（已完成 {len(done_q)}）===", flush=True)
    out = open(ap, "a", encoding="utf-8")
    t0 = time.time()
    done = [0]

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
        with LOCK:
            out.write(json.dumps({"qid": c["qid"], "query": c["query"], "gold": c["gold"],
                                  "category": c.get("category"), "facts_n": c.get("facts_n"),
                                  "answer_obj": obj}, ensure_ascii=False) + "\n")
            done[0] += 1
            if done[0] % 100 == 0:
                print(f"  作答 {done[0]}/{len(todo_recs)} | {time.time()-t0:.0f}s", flush=True)

    with ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(work, todo_recs))
    out.close()

    ans = [json.loads(l) for l in open(ap, encoding="utf-8")]
    vp = P(f"verdicts_707_{mode}.jsonl")
    done_v = set()
    if os.path.exists(vp):
        for line in open(vp, encoding="utf-8"):
            try:
                done_v.add(json.loads(line)["qid"])
            except Exception:
                pass
    ans = [a for a in ans if a["qid"] not in done_v]
    print(f"  待判 {len(ans)}（已判 {len(done_v)}）", flush=True)
    t1 = time.time()
    vout = open(vp, "a", encoding="utf-8")
    cnt = Counter()

    def wj(a):
        v = judge(a["query"], a["gold"], str(a["answer_obj"].get("answer") or ""))
        with LOCK:
            vout.write(json.dumps({"qid": a["qid"], "verdict": v}, ensure_ascii=False) + "\n")
            cnt[v] += 1

    with ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(wj, ans))
    vout.close()
    n = len(ans)
    print(f"  判分 {time.time()-t1:.0f}s | correct {cnt['correct']/n:.1%} | "
          f"+partial {(cnt['correct']+cnt['partial'])/n:.1%} | wrong {cnt['wrong']/n:.1%} | "
          f"refused {cnt['refused']/n:.1%}", flush=True)

    common = [a["qid"] for a in ans if a["qid"] in base]
    if common:
        cur = Counter(base[q] for q in common)
        m = len(common)
        print(f"  同批现状（旧管线）：correct {cur['correct']/m:.1%} | "
              f"+partial {(cur['correct']+cur['partial'])/m:.1%} | wrong {cur['wrong']/m:.1%} | "
              f"refused {cur['refused']/m:.1%}   (n={m})", flush=True)
