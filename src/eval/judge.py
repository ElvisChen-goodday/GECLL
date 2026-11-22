"""重判 answers_300.json（上次判分因 finder_rag 没有 llm() 全部报错）"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
from config import P, DATA_DIR, ENV_FILE, load_env
import os, re, json, time
from collections import Counter, defaultdict
import requests


env = {}
env = load_env()
BASE = env.get("OPENAI_BASE_URL", "https://api.deepseek.com/v1").rstrip("/")
KEY = env.get("OPENAI_API_KEY", "")
MODEL = env.get("OPENAI_MODEL", "deepseek-chat")

JUDGE = ("你是严格的财务问答评审。判断「模型答案」是否回答了「问题」，以「金标答案」为准。\n"
         "correct = 关键事实/数值与金标一致（表述不同但实质相同也算）；"
         "partial = 命中部分要点但缺关键信息或口径不同；wrong = 事实错误；refused = 拒答。\n"
         '只输出 JSON：{"verdict": "correct|partial|wrong|refused"}\n\n')


def judge(q, gold, ans):
    prompt = (JUDGE + "问题：" + str(q) + "\n金标答案：" + str(gold)[:1500] +
              "\n模型答案：" + str(ans)[:1500])
    try:
        r = requests.post(BASE + "/chat/completions", timeout=90,
                          headers={"Authorization": "Bearer " + KEY},
                          json={"model": MODEL, "temperature": 0,
                                "messages": [{"role": "user", "content": prompt}]})
        r.raise_for_status()
        txt = r.json()["choices"][0]["message"]["content"]
        m = re.search(r"\{.*\}", txt, re.S)
        return json.loads(m.group(0)).get("verdict", "?") if m else "?"
    except Exception as e:
        print("   judge 异常:", type(e).__name__, str(e)[:120], flush=True)
        return "error"


rows = json.load(open(P("answers_300.json"), encoding="utf-8"))
cnt = Counter()
bycat = defaultdict(Counter)
t0 = time.time()
for i, rec in enumerate(rows, 1):
    o = rec["answer_obj"]
    ans = str(o.get("answer") or "") or str(o.get("best_effort") or "")
    v = judge(rec["query"], rec["gold"], ans)
    rec["verdict"] = v
    cnt[v] += 1
    bycat[rec.get("category") or "?"][v] += 1
    if i % 25 == 0:
        print(f"  {i}/{len(rows)} | {dict(cnt)} | {time.time()-t0:.0f}s", flush=True)

json.dump(rows, open(P("answers_300_judged.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
n = len(rows)
ok, pa = cnt["correct"], cnt["partial"]
print(f"\n【答案级评测】{n} 题（type=None、非表题、有金标）")
print(f"  correct        {ok:>4}/{n} = {ok/n:.0%}")
print(f"  correct+partial{ok+pa:>4}/{n} = {(ok+pa)/n:.0%}")
print(f"  wrong {cnt['wrong']:>3} | refused {cnt['refused']:>3} | 解析失败 {cnt['?']+cnt['error']:>3}")
print(f"  平均上下文 {sum(r['ctx_chars'] for r in rows)/n:,.0f} 字符")
print("\n分类别（correct+partial）：")
for k, c in sorted(bycat.items(), key=lambda kv: -sum(kv[1].values())):
    tot = sum(c.values())
    print(f"  {k:<22}{c['correct']+c['partial']:>4}/{tot:<4} = {(c['correct']+c['partial'])/tot:>3.0%}"
          f"   (correct {c['correct']})")
