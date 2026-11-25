"""结构感知 + 父子 chunk（clean 重写版）

结构信号：<h1>~<h6>、`Item 7.` / `PART II` / `NOTE 5.` 这类编号标题、全大写短语
噪声处理：同文档内重复出现（数字归一后）≥4 次的短文本视为页眉页脚，直接丢弃
父子关系：
  parent = 一个章节（≤4000 字符，按段落累积）；表格 = 整表一个 parent
  child  = parent 内 600 字符级、句界对齐、10% 句子重叠；表格 = 表头 + 5 行
只 embed child；命中后用 parent_id 取回整节/整表作为上下文。

字段：chunk_id, doc_id, level(parent|child), kind(text|table), parent_id,
      section_path, text, n_chars, table_id, n_rows, n_cols, row_from, row_to
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
from config import P, DATA_DIR, ENV_FILE, load_env
import os, re, json, glob, time, argparse
from concurrent.futures import ProcessPoolExecutor
import importlib.util

from lxml import etree


from src.parse import parallel_chunks_v2 as pc

BLOCK = {"h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "div"}
HEAD_RE = re.compile(r"^(item\s+\d+[a-c]?[.:]?|part\s+[ivx]+\.?|note\s+\d+[.:]?|article\s+[ivx\d]+)", re.I)
CAPS_RE = re.compile(r"^[A-Z0-9 ,&/'’\-\.()]{6,90}$")
SENT_RE = re.compile(r"(?<=[.!?;])\s+")
JUNK_HEAD_RE = re.compile(
    r"^(united states( securities and exchange commission)?|securities and exchange commission|"
    r"washington,?\s*d\.?\s*c\.?(\s*\d+)?|form\s+10-k|documents incorporated by reference|"
    r"table of contents)\s*$", re.I)

PARENT_CHARS = 4000
PARENT_MAX = 6000
CHILD_CHARS = 600
CHILD_OVERLAP = 0.10
CHILD_MIN = 200
CHILD_MAX = 1200
TABLE_PARENT_CHARS = 6000
TABLE_CHILD_ROWS = 5
MIN_SECTION = 600


def is_heading(tag, text):
    if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
        return True, int(tag[1])
    t = text.strip()
    if not (4 <= len(t) <= 120) or JUNK_HEAD_RE.match(t):
        return False, 0
    m = HEAD_RE.match(t)
    if m:
        return True, 1 if m.group(1).lower().startswith("part") else 2
    if CAPS_RE.match(t) and len(t.split()) >= 2:
        return True, 3
    return False, 0


def parse_blocks(path):
    tree = etree.fromstring(open(path, "rb").read(), etree.HTMLParser(recover=True, huge_tree=True))
    for el in [e for e in tree.iter() if isinstance(e.tag, str) and pc.local(e) in ("header", "hidden")]:
        p = el.getparent()
        if p is not None:
            p.remove(el)

    blocks = []
    for el in tree.iter():
        if not isinstance(el.tag, str):
            continue
        tag = pc.local(el)
        if tag in BLOCK:
            if any(pc.local(c) in BLOCK | {"table"} for c in el if isinstance(c.tag, str)):
                continue
            if any(pc.local(x) == "table" for x in el.iter() if isinstance(x.tag, str)):
                continue
            t = pc.text_of(el)
            if len(t) < 25:
                continue
            hd, lvl = is_heading(tag, t)
            blocks.append(("heading", lvl, t[:160]) if hd else ("text", 0, t))
        elif tag == "table":
            p, nested = el.getparent(), False
            while p is not None:
                if isinstance(p.tag, str) and pc.local(p) == "table":
                    nested = True
                    break
                p = p.getparent()
            if nested:
                continue
            trs = [c for c in el.iter() if isinstance(c.tag, str) and pc.local(c) == "tr"]
            if len(trs) < 2:
                continue
            grid = pc.expand_grid(trs)
            if pc.table_ok(grid):
                blocks.append(("table", 0, grid))
            else:
                flat = "\n".join(pc.row_text(r) for r in grid if any(r))
                if len(flat) >= 40:
                    blocks.append(("text", 0, flat))

    # 页眉页脚：数字归一后重复 ≥4 次的短文本
    def fkey(t):
        return re.sub(r"\d+", "#", re.sub(r"\s+", " ", t).strip().lower())

    freq = {}
    for b in blocks:
        t = b[2]
        if isinstance(t, str) and len(t) <= 120:
            freq[fkey(t)] = freq.get(fkey(t), 0) + 1
    return [b for b in blocks
            if not (isinstance(b[2], str) and len(b[2]) <= 120 and freq.get(fkey(b[2]), 0) >= 4)]


def split_children(text):
    sents = [s for s in SENT_RE.split(text) if s.strip()]
    if not sents:
        return []
    parts, buf, size = [], [], 0
    for s in sents:
        if size + len(s) > CHILD_CHARS and buf:
            parts.append(" ".join(buf).strip())
            keep, ksize = [], 0
            for prev in reversed(buf):
                if ksize + len(prev) > CHILD_CHARS * CHILD_OVERLAP:
                    break
                keep.insert(0, prev)
                ksize += len(prev)
            buf, size = keep[:], ksize
        buf.append(s)
        size += len(s)
    if buf:
        parts.append(" ".join(buf).strip())

    # 过短并进前块；过长按词硬切
    merged = []
    for p in parts:
        if merged and len(p) < CHILD_MIN:
            merged[-1] = (merged[-1] + " " + p).strip()
        else:
            merged.append(p)
    out = []
    for p in merged:
        while len(p) > CHILD_MAX:
            cut = p.rfind(" ", 0, CHILD_MAX)
            cut = cut if cut > CHILD_MAX * 0.5 else CHILD_MAX
            out.append(p[:cut].strip())
            p = p[cut:].strip()
        if p:
            out.append(p)
    final = []
    for p in out:
        if final and len(p) < CHILD_MIN and len(final[-1]) + len(p) < CHILD_MAX:
            final[-1] = (final[-1] + " " + p).strip()
        else:
            final.append(p)
    return final


def chunk_doc(path):
    doc_id = os.path.basename(path)[:-5]
    blocks = parse_blocks(path)

    out, stack, buf = [], [], []
    state = {"p": 0, "t": 0}

    def path_str():
        return " > ".join(t for _, t in stack) or "(document opening)"

    def new_parent(text, kind):
        if kind == "text":
            state["p"] += 1
            pid = f"{doc_id}:P{state['p']:04d}"
        else:
            state["t"] += 1
            pid = f"{doc_id}:T{state['t']:04d}"
        out.append({"chunk_id": pid, "doc_id": doc_id, "level": "parent", "kind": kind,
                    "parent_id": "", "section_path": path_str(), "text": text,
                    "n_chars": len(text)})
        return pid

    def flush():
        nonlocal buf
        if not buf:
            return
        text = "\n".join(buf)
        # 超大块按 6000 字符拆成多个 parent，避免 10 万字符的 parent
        segs = [text]
        if len(text) > PARENT_MAX:
            segs, cur = [], ""
            for piece in split_children(text):
                if cur and len(cur) + len(piece) + 1 > PARENT_MAX:
                    segs.append(cur)
                    cur = ""
                cur = (cur + " " + piece).strip() if cur else piece
            if cur:
                segs.append(cur)
        for seg in segs:
            pid = new_parent(seg, "text")
            for i, c in enumerate(split_children(seg), 1):
                out.append({"chunk_id": f"{pid}-c{i:02d}", "doc_id": doc_id, "level": "child",
                            "kind": "text", "parent_id": pid, "section_path": path_str(),
                            "text": c, "n_chars": len(c)})
        buf = []

    for kind, lvl, payload in blocks:
        if kind == "heading":
            flush()
            while stack and stack[-1][0] >= lvl:
                stack.pop()
            stack.append((lvl, payload))
        elif kind == "text":
            buf.append(payload)
            if sum(len(x) + 1 for x in buf) >= PARENT_CHARS:
                flush()
        else:
            grid = payload
            lead = ""
            if buf and sum(len(x) + 1 for x in buf) < MIN_SECTION:
                lead = "\n".join(buf)        # 表前引导语并入表格
                buf = []
            else:
                flush()
            # 表头去重：rowspan 展开后首两行经常完全相同
            header, seen = [], set()
            for r in grid[:3]:
                if not any(r):
                    continue
                t = pc.row_text(r).strip()
                if t and t not in seen:
                    seen.add(t)
                    header.append(r)
                if len(header) >= 2:
                    break
            body = grid[len(header):] if header else grid
            # 去掉 rowspan 展开产生的相邻重复行
            ded, prev = [], None
            for r in body:
                t = pc.row_text(r).strip()
                if t and t != prev:
                    ded.append(r)
                prev = t
            body = ded
            head_txt = "\n".join(pc.row_text(r) for r in header)

            # 表标题：优先取表前引导语的最后一句，退化为最近一级标题
            cap = ""
            for ln in reversed([x.strip() for x in lead.split("\n") if x.strip()]):
                if 8 <= len(ln) <= 220:
                    cap = ln
                    break
            if not cap and stack:
                cap = stack[-1][1][:220]

            # 表内科目：首列前 12 个非纯数字标签，给每个 child 提供"这张表在讲什么"
            labels = []
            for r in body:
                cell = ""
                for c0 in r:
                    if c0 and c0.strip():
                        cell = c0.strip()
                        break
                if 2 <= len(cell) <= 90 and not re.fullmatch(r"[\$\(\)\d,\.%\-\s]+", cell):
                    if cell not in labels:
                        labels.append(cell)
                if len(labels) >= 12:
                    break
            prefix = ""
            if stack:
                sp = " > ".join(t for _, t in stack[-2:])
                prefix += f"[章节] {sp[:140]}\n"
            if cap:
                prefix += f"[表] {cap}\n"
            if labels:
                prefix += "[表内科目] " + " ; ".join(labels) + "\n"
            prefix = prefix[:460]

            full = (lead + "\n" + head_txt + "\n" + "\n".join(pc.row_text(r) for r in body)).strip()
            tpid = new_parent(full[:TABLE_PARENT_CHARS], "table")
            out[-1].update({"table_id": state["t"], "n_rows": len(grid),
                            "n_cols": max((len(r) for r in grid), default=0)})
            for gi in range(0, len(body), TABLE_CHILD_ROWS):
                grp = body[gi:gi + TABLE_CHILD_ROWS]
                txt = (prefix + head_txt + "\n" + "\n".join(pc.row_text(r) for r in grp)).strip()
                if len(txt) < 30:
                    continue
                out.append({"chunk_id": f"{tpid}-r{gi // TABLE_CHILD_ROWS + 1:03d}", "doc_id": doc_id,
                            "level": "child", "kind": "table", "parent_id": tpid,
                            "section_path": path_str(), "text": txt, "n_chars": len(txt),
                            "table_id": state["t"], "n_rows": len(grp),
                            "n_cols": max((len(r) for r in grid), default=0),
                            "row_from": gi, "row_to": gi + len(grp) - 1})
    flush()

    for c in out:
        if not c.get("section_path"):
            c["section_path"] = "(document opening)"
    return doc_id, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="chunks_struct.jsonl")
    ap.add_argument("--done", default=".done_struct")
    a = ap.parse_args()

    files = sorted(glob.glob("10k/*.html"))[: a.limit or None]
    done = set()
    if os.path.exists(a.done):
        done = {l.strip() for l in open(a.done, encoding="utf-8") if l.strip()}
    todo = [f for f in files if os.path.basename(f)[:-5] not in done]
    print(f"文件 {len(files)} | 已完成 {len(done)} | 本轮 {len(todo)} | 进程 {a.workers}", flush=True)
    if not todo:
        return

    t0 = time.time()
    cnt = {"parent": 0, "child": 0}
    kcnt = {"text": 0, "table": 0}
    with open(a.out, "a" if done else "w", encoding="utf-8") as f, \
         open(a.done, "a", encoding="utf-8") as d:
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            for i, (doc, chunks) in enumerate(ex.map(chunk_doc, todo, chunksize=1), 1):
                for c in chunks:
                    if c["level"] == "child" and c["n_chars"] < 40:
                        continue          # 过短碎片（"income." 这类），靠 parent 兜底
                    cnt[c["level"]] += 1
                    kcnt[c["kind"]] += 1
                    f.write(json.dumps(c, ensure_ascii=False) + "\n")
                f.flush()
                d.write(doc + "\n")
                d.flush()
                if i % 50 == 0 or i == len(todo):
                    el = time.time() - t0
                    print(f"[{i}/{len(todo)}] {el:5.1f}s ETA {el / i * (len(todo) - i):5.1f}s "
                          f"parent {cnt['parent']:,} child {cnt['child']:,}", flush=True)
    print(f"\n完成 {len(todo)} 份 / {time.time() - t0:.1f}s | parent {cnt['parent']:,} "
          f"child {cnt['child']:,} | text {kcnt['text']:,} table {kcnt['table']:,} "
          f"| {os.path.getsize(a.out) / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
