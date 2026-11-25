"""
497 份 10-K HTML → chunks.jsonl（多进程，文件级并行）

修复的两个坑（v1 的问题）：
1. iXBRL 头没跳干净：ix:header 里有 references/resources，schemaRef 会拼成
   "http://fasb.org/us-gaap/..." 的垃圾文本块。必须按"元素对象集合"过滤整棵子树
   （用 id() 不行，lxml 代理对象会被回收复用）。
2. 表格 chunk_id 重复：拼装阶段误用了循环结束后的 table_no，导致所有表格同 id。

用法：
    python parallel_chunks_v2.py --limit 20 --out chunks_trial.jsonl --done .done_trial
    python parallel_chunks_v2.py --workers 10
"""
import os, re, json, glob, time, argparse
from concurrent.futures import ProcessPoolExecutor

WS = re.compile(r"\s+")
DIGIT = re.compile(r"\d")
BLOCK = {"h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "div"}

CHUNK_CHARS = 1200
OVERLAP = 0.15
TABLE_MAX = 6000
TABLE_GROUP_ROWS = 25


def local(el):
    t = el.tag
    if not isinstance(t, str):
        return ""
    if "}" in t:
        t = t.split("}", 1)[1]
    return t.split(":")[-1].lower()


INVISIBLE = dict.fromkeys(map(ord, "\u200b\u200c\u200d\u2060\ufeff\u00a0"), " ")


def text_of(el):
    return WS.sub(" ", "".join(el.itertext()).translate(INVISIBLE)).strip()


def kids(el):
    return [c for c in el if isinstance(c.tag, str)]


def skip_set(tree):
    skip = set()
    for el in tree.iter():
        if isinstance(el.tag, str) and local(el) in ("header", "hidden"):
            skip.add(el)
            skip.update(x for x in el.iter() if isinstance(x.tag, str))
    return skip


def expand_grid(rows):
    grid, occupied = [], set()
    for r, tr in enumerate(rows):
        cells = [c for c in kids(tr) if local(c) in ("td", "th")]
        row, col = [], 0
        for cell in cells:
            while (r, col) in occupied:
                row.append(None)
                col += 1
            try:
                span = max(1, min(int(cell.get("colspan") or 1), 30))
            except ValueError:
                span = 1
            try:
                rspan = max(1, min(int(cell.get("rowspan") or 1), 30))
            except ValueError:
                rspan = 1
            for dr in range(rspan):
                for dc in range(span):
                    occupied.add((r + dr, col + dc))
            row.append(text_of(cell))
            row.extend([None] * (span - 1))
            col += span
        grid.append(row)
    width = max((len(x) for x in grid), default=0)
    return [x + [None] * (width - len(x)) for x in grid]


def merge_cells(row):
    """把单独成格的货币符号/括号并入下一个数字格，并丢掉空格。
    "$ | 757" → "$757"；这样统计数字密度时不会被 "$" 稀释。"""
    cells = [c for c in ("" if x is None else x.strip() for x in row) if c]
    out, pend = [], ""
    for c in cells:
        if c in ("$", "€", "£", "¥", "%", "(", "((("):
            pend += c
            continue
        out.append(pend + c)
        pend = ""
    if pend:
        out.append(pend)
    return out


def table_ok(grid):
    if len(grid) < 2:
        return False
    # 表头行（"2023 | 2022 | 2021"、"(In millions)"）会稀释数字密度，
    # 分母只算表体：超过 6 行时跳过前 2 行
    body = grid[2:] if len(grid) > 6 else grid
    cells = [c for r in body for c in merge_cells(r)]
    if len(cells) < 6:
        return False
    all_cells = [c for r in grid for c in merge_cells(r)]
    num_cells = [c for c in all_cells if DIGIT.search(c) and len(c) <= 40]
    # 有些表把整行数字塞进一个单元格（colspan 合并），按"数字个数"而不是"数字单元格数"计数
    num_tokens = sum(len(re.findall(r"\d[\d,]*(?:\.\d+)?", c)) for c in all_cells)
    min_num = 4 if len(grid) <= 3 else 8
    if num_tokens < min_num or len(num_cells) / max(1, len(cells)) < 0.06:
        return False
    numeric = num_cells
    # 目录表：表头出现 Page / Item N / Contents，且数字基本都是小整数（页码）
    head = " ".join(c for r in grid[:6] for c in merge_cells(r))[:400]
    if re.search(r"(page|item|contents|index)", head, re.I):
        vals = []
        for c in numeric:
            try:
                vals.append(float(c.replace(",", "").replace("$", "").strip("()%")))
            except ValueError:
                pass
        if vals and sum(1 for v in vals if v < 400) / len(vals) >= 0.6:
            return False
    width = max((len(r) for r in grid), default=0)
    cols_num = 0
    nonmono_single = False
    col_min = 1 if len(grid) <= 3 else 3      # 2~3 行的表，一列只有 1~2 个数字也算
    for c in range(width):
        col = [r[c] for r in grid if c < len(r) and r[c] and DIGIT.search(r[c]) and len(r[c]) <= 40]
        if len(col) >= col_min:
            cols_num += 1
            vals = []
            for x in col:
                try:
                    vals.append(float(x.replace(",", "").replace("$", "").strip("()%")))
                except ValueError:
                    vals = []
                    break
            # 单列数据表（州→门店数、设施→床位）：值非单调递增就不像目录页码
            if len(vals) >= 10 and any(vals[i] < vals[i - 1] for i in range(1, len(vals))):
                nonmono_single = True
    return cols_num >= 2 or (cols_num >= 1 and nonmono_single)


def row_text(row):
    return " | ".join(merge_cells(row))


def split_text(text, limit):
    out, step = [], int(limit * (1 - OVERLAP))
    i, n = 0, len(text)
    while i < n:
        j = min(n, i + limit)
        if j < n:
            cut = max(text.rfind(". ", i + step // 2, j), text.rfind(" ", i + step // 2, j))
            if cut > i + step // 2:
                j = cut + 1
        out.append(text[i:j].strip())
        if j >= n:
            break
        i = max(j - int(limit * OVERLAP), i + 1)
    return [c for c in out if c]


def chunk_file(path):
    doc_id = os.path.basename(path)[:-5]
    from lxml import etree
    tree = etree.fromstring(open(path, "rb").read(), etree.HTMLParser(recover=True, huge_tree=True))
    # 直接从树上摘掉 iXBRL 头（ix:header / ix:hidden），而不是"遍历时跳过"：
    # 否则包含它的外层容器也会被连带跳过，正文会整块丢失（HSIC/GIS 实测掉 25%+）
    for el in [e for e in tree.iter() if isinstance(e.tag, str) and local(e) in ("header", "hidden")]:
        p = el.getparent()
        if p is not None:
            p.remove(el)
    skip = set()

    pieces, heading, tno = [], "", 0
    for el in tree.iter():
        if not isinstance(el.tag, str) or el in skip:
            continue
        tag = local(el)
        if tag in BLOCK:
            if any(local(c) in BLOCK | {"table"} for c in kids(el)):
                continue
            # 表格若是"孙子"级（<div><div><table>），容器同样要跳过，
            # 否则 itertext 会把整张表拍平成文本，表格就消失了
            if any(local(x) == "table" for x in el.iter() if isinstance(x.tag, str)):
                continue
            t = text_of(el)
            if len(t) < 30:
                continue
            if tag.startswith("h") and len(t) <= 160:
                heading = t
                continue
            pieces.append(("text", heading, t))
        elif tag == "table":
            p = el.getparent()
            nested = False
            while p is not None:
                if isinstance(p.tag, str) and local(p) == "table":
                    nested = True
                    break
                p = p.getparent()
            if nested:
                continue
            trs = [c for c in el.iter() if isinstance(c.tag, str) and local(c) == "tr"]
            if len(trs) < 2:
                continue
            grid = expand_grid(trs)
            if not table_ok(grid):
                # 版式表 / 叙述型表：不当作表格 chunk，但文本不能丢，
                # 否则像 HSIC/GIS 这种文档会掉 20%+ 正文（实测）
                flat = "\n".join(row_text(r) for r in grid if any(r))
                for part in split_text(flat, CHUNK_CHARS):
                    if len(part) >= 40:
                        pieces.append(("text", heading, part))
                continue
            tno += 1
            pieces.append(("table", heading, (tno, grid)))

    chunks, buf, buf_heading = [], [], ""

    def flush():
        if not buf:
            return
        for part in split_text("\n".join(buf), CHUNK_CHARS):
            chunks.append({"chunk_id": f"{doc_id}:t{len(chunks)+1:04d}", "doc_id": doc_id,
                           "kind": "text", "section": buf_heading, "text": part, "n_chars": len(part)})
        buf.clear()

    for kind, hd, payload in pieces:
        if kind == "text":
            if not buf:
                buf_heading = hd
            elif hd != buf_heading or sum(len(x) + 1 for x in buf) >= CHUNK_CHARS:
                flush()
                buf_heading = hd
            buf.append(payload)
            if sum(len(x) + 1 for x in buf) >= CHUNK_CHARS:
                flush()
            continue

        flush()
        tid, grid = payload
        header = [r for r in grid[:2] if any(r)]
        body = grid[len(header):] if header else grid
        groups = [body[i:i + TABLE_GROUP_ROWS] for i in range(0, len(body), TABLE_GROUP_ROWS)] or [[]]
        head_txt = "\n".join(row_text(r) for r in header)
        for gi, grp in enumerate(groups, 1):
            body_txt = "\n".join(row_text(r) for r in grp if any(r))
            txt = (head_txt + "\n" + body_txt).strip()
            if len(txt) < 40:
                continue
            cid = f"{doc_id}:tab{tid:03d}" + (f"-p{gi}" if len(groups) > 1 else "")
            chunks.append({"chunk_id": cid, "doc_id": doc_id, "kind": "table", "section": hd,
                           "table_id": tid, "n_rows": len(grp),
                           "n_cols": len(grid[0]) if grid else 0,
                           "text": txt[:TABLE_MAX], "n_chars": min(len(txt), TABLE_MAX)})
    flush()
    return doc_id, chunks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="chunks.jsonl")
    ap.add_argument("--done", default=".chunks_done")
    a = ap.parse_args()

    files = sorted(glob.glob("10k/*.html"))[: a.limit or None]
    done = set()
    if os.path.exists(a.done):
        done = {l.strip() for l in open(a.done, encoding="utf-8") if l.strip()}
    todo = [f for f in files if os.path.basename(f)[:-5] not in done]
    print(f"文件 {len(files)} | 已完成 {len(done)} | 本轮 {len(todo)} | 进程 {a.workers}", flush=True)
    if not todo:
        return

    t0, n, n_text, n_tab = time.time(), 0, 0, 0
    with open(a.out, "a" if done else "w", encoding="utf-8") as f, \
         open(a.done, "a", encoding="utf-8") as d:
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            for i, (doc_id, chunks) in enumerate(ex.map(chunk_file, todo, chunksize=1), 1):
                for c in chunks:
                    f.write(json.dumps(c, ensure_ascii=False) + "\n")
                f.flush()
                d.write(doc_id + "\n")
                d.flush()
                n += len(chunks)
                n_text += sum(1 for c in chunks if c["kind"] == "text")
                n_tab += sum(1 for c in chunks if c["kind"] == "table")
                if i % 25 == 0 or i == len(todo):
                    el = time.time() - t0
                    print(f"[{i}/{len(todo)}] {el:6.1f}s ETA {el/i*(len(todo)-i):5.1f}s "
                          f"chunks {n:,} (text {n_text:,} / table {n_tab:,})", flush=True)
    print(f"\n完成 {len(todo)} 份 / {time.time()-t0:.1f}s | chunks {n:,} "
          f"(text {n_text:,} / table {n_tab:,}) | {os.path.getsize(a.out)/1e6:.1f} MB")


if __name__ == "__main__":
    main()
