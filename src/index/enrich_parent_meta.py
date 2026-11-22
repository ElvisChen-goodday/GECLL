"""离线生成 parent 元数据：passage_role / neighbor_ids / has_year / years / 数字密度"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
from config import P, DATA_DIR, ENV_FILE, load_env
import json, re


YEAR = re.compile(r"\b((?:19|20)\d{2})\b")
NUMD = re.compile(r"\d")
BOILER = re.compile(r"(forward[- ]looking|risk factor|legal proceedings|controls and procedures|"
                    r"accounting polic|critical accounting|safe harbor)", re.I)
NOTE = re.compile(r"(^|\s)(note\s+\d+|\(\d+\)|see note)", re.I)
LEAD = re.compile(r"(the following table|following tables|summariz|set forth below|as follows|"
                  r"the table below|presented below)", re.I)


def role_of(sec, txt):
    s = (sec or "").lower()
    if BOILER.search(s) or BOILER.search(txt[:200]):
        return "boilerplate"
    if LEAD.search(txt[:400]):
        return "table_lead"
    dens = sum(1 for c in txt if NUMD.match(c)) / max(len(txt), 1)
    if dens > 0.12:
        return "numeric_dense"
    if re.search(r"\bitem\s+\d+", s) and "financial statements" in s:
        return "fs_note"
    if NOTE.search(txt[:300]):
        return "fs_note"
    if len(txt) < 400:
        return "short_fragment"
    return "narrative"


per_doc = {}
parents = []
with open(P("chunks_struct.jsonl"), encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        if r["level"] == "parent" and r["kind"] == "text":
            parents.append(r)
            per_doc.setdefault(r["doc_id"], []).append(r["chunk_id"])

order = {}
for d, ids in per_doc.items():
    ids = sorted(ids)
    for i, pid in enumerate(ids):
        order[pid] = (ids[i - 1] if i > 0 else None, ids[i + 1] if i + 1 < len(ids) else None)

with open(P("parent_meta.jsonl"), "w", encoding="utf-8") as out:
    for r in parents:
        txt = r["text"]
        prev, nxt = order.get(r["chunk_id"], (None, None))
        years = sorted(set(YEAR.findall(txt)))
        out.write(json.dumps({
            "chunk_id": r["chunk_id"], "doc_id": r["doc_id"],
            "section_path": r.get("section_path") or "",
            "role": role_of(r.get("section_path", ""), txt),
            "prev_id": prev, "next_id": nxt,
            "has_year": bool(years), "years": years[:6], "has_number": bool(NUMD.search(txt)),
            "num_density": round(sum(1 for c in txt if NUMD.match(c)) / max(len(txt), 1), 3),
            "n_chars": len(txt)}, ensure_ascii=False) + "\n")

print(f"parent_meta.jsonl 写出 {len(parents):,} 段")
from collections import Counter
c = Counter(role_of(r.get("section_path", ""), r["text"]) for r in parents)
for k, v in c.most_common():
    print(f"  {k:<18}{v:>8,}  {v/len(parents):>6.1%}")
