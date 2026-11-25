"""上下文组装（全文版）：把命中的 top-5 parent 整段送入，而不是骨架+3 句

依据：800 题分层验证，文本题 correct 18.4% → 21.4%（+3.0pp），拒答与错误率不变。
表特征题不走这条路径（用表渲染旁路），因此默认跳过 tbl=True。

用法：
  python scripts/build_ctx_full.py                # 全部文本题
  python scripts/build_ctx_full.py --category Governance
  python scripts/build_ctx_full.py --limit 200    # 试跑
可续跑：已写入的 qid 自动跳过。
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from config import P, DATA_DIR, ENV_FILE, load_env

import json, argparse, os

PER_PARENT = 2800     # 每段最多送多少字符
BUDGET = 14000        # 单题上下文上限
OUT = P("ctx_text_full.jsonl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--include-tables", action="store_true")
    args = ap.parse_args()

    recs = []
    for line in open(P("ctx_all.jsonl"), encoding="utf-8"):
        r = json.loads(line)
        if not args.include_tables and r.get("tbl"):
            continue
        if args.category and (r.get("category") or "") != args.category:
            continue
        recs.append(r)

    done = set()
    if os.path.exists(OUT):
        for line in open(OUT, encoding="utf-8"):
            try:
                done.add(json.loads(line)["qid"])
            except Exception:
                pass
    todo = [r for r in recs if r["qid"] not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"候选 {len(recs)} | 已完成 {len(done)} | 本轮 {len(todo)}", flush=True)
    if not todo:
        print("ALLDONE")
        return

    need = {p for r in todo for p in (r.get("parents") or [])}
    print(f"需要 parent 文本 {len(need):,}", flush=True)
    ptext = {}
    for line in open(P("chunks_struct.jsonl"), encoding="utf-8"):
        if not need:
            break
        r = json.loads(line)
        if r["chunk_id"] in need:
            ptext[r["chunk_id"]] = r["text"]
            need.discard(r["chunk_id"])
    print(f"读到 {len(ptext):,} 段", flush=True)

    with open(OUT, "a", encoding="utf-8") as f:
        for i, r in enumerate(todo, 1):
            parts, used = [], 0
            for n, pid in enumerate((r.get("parents") or [])[:5], 1):
                t = ptext.get(pid, "")
                if not t:
                    continue
                seg = t[:PER_PARENT]
                if used + len(seg) > BUDGET:
                    seg = seg[: max(0, BUDGET - used)]
                if not seg:
                    break
                parts.append(f"[{n}] {pid}\n{seg}")
                used += len(seg)
            ctx = "【章节全文（按相关性排序）】\n" + "\n\n".join(parts)
            f.write(json.dumps({**r, "ctx": ctx, "facts_n": 0}, ensure_ascii=False) + "\n")
            if i % 300 == 0:
                print(f"  {i}/{len(todo)}", flush=True)
    print(f"完成 → {OUT}", flush=True)


if __name__ == "__main__":
    main()
