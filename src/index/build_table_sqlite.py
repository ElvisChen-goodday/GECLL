"""v3：给每个单元格补上「年份」与「行标签」，解决"第 3 列是几几年"的问题

做法：
1. 展开表格时保留 colspan 信息（span_map），合并表头的年份可以覆盖到整段列
2. 表头带（前 8 行）里找年份行：2023 / FY2024 / December 31, 2023
   一个单元格里写 "2023 2022 2021" 时，按顺序分配给后面的列
3. 行标签：该行左侧第一个非数字、≥3 个字母的单元格；取不到就继承上一行（多行标签）
4. 写库：cells 增加 year / row_label 两列，另建 column_map 便于核查

输出：finder_tables.db（覆盖 v2 的同名库）
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
from config import P, DATA_DIR, ENV_FILE, load_env
import os, re, csv, glob, time, sqlite3, argparse
from concurrent.futures import ProcessPoolExecutor
import importlib.util
from lxml import etree


spec = importlib.util.spec_from_file_location("pc", "parallel_chunks_v2.py")
pc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pc)

NUMRE = re.compile(r"^\(?-?\$?\s?[\d,]+(?:\.\d+)?\)?%?$")
YEARRE = re.compile(r"(?<!\d)((?:FY|fiscal year )?(?:19|20)\d{2})(?!\d)", re.I)
BLOCK = {"h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "div"}
JUNK_HEADING = re.compile(
    r"^(united states|securities and exchange commission|washington,? ?d\.?c|form 10-k|"
    r"table of contents|part [ivx]+|item \d+[a-c]?\.?$)", re.I)

SCHEMA = [
    "PRAGMA journal_mode=WAL", "PRAGMA synchronous=OFF",
    "DROP TABLE IF EXISTS cells", "DROP TABLE IF EXISTS tables",
    "DROP TABLE IF EXISTS tables_fts", "DROP TABLE IF EXISTS docs",
    "DROP TABLE IF EXISTS column_map",
    """CREATE TABLE docs(doc_id TEXT PRIMARY KEY, ticker TEXT, company TEXT,
       fiscal_year TEXT, cik TEXT, file TEXT)""",
    """CREATE TABLE tables(table_pk INTEGER PRIMARY KEY, doc_id TEXT NOT NULL,
       table_idx INTEGER NOT NULL, section TEXT, caption TEXT, text TEXT,
       n_rows INTEGER, n_cols INTEGER, n_cells INTEGER, n_numeric INTEGER,
       n_year_cols INTEGER, src_idx INTEGER)""",
    """CREATE TABLE cells(table_pk INTEGER NOT NULL, row INTEGER NOT NULL,
       col INTEGER NOT NULL, text TEXT, value REAL, year TEXT, row_label TEXT)""",
    "CREATE VIRTUAL TABLE tables_fts USING fts5(text, doc_id UNINDEXED, table_pk UNINDEXED)",
    """CREATE TABLE column_map(table_pk INTEGER, col INTEGER, year TEXT,
       header_text TEXT)""",
]


def to_value(t):
    if not t:
        return None
    s = t.strip()
    if not NUMRE.match(s):
        return None
    s2 = s.replace("$", "").replace(",", "").replace("%", "")
    neg = s2.startswith("(") and s2.endswith(")")
    s2 = s2.strip("()")
    try:
        v = float(s2)
    except ValueError:
        return None
    return -v if neg else v


def grid_with_spans(rows):
    """返回 (grid, spans)：spans[r] = [(col_start, span, text), ...]"""
    grid, spans, occ = [], [], set()
    for r, tr in enumerate(rows):
        cells = [c for c in tr if isinstance(c.tag, str) and pc.local(c) in ("td", "th")]
        row, sp, col = [], [], 0
        for cell in cells:
            while (r, col) in occ:
                row.append(None)
                col += 1
            try:
                cs = max(1, min(int(cell.get("colspan") or 1), 30))
            except ValueError:
                cs = 1
            try:
                rs = max(1, min(int(cell.get("rowspan") or 1), 30))
            except ValueError:
                rs = 1
            for dr in range(rs):
                for dc in range(cs):
                    occ.add((r + dr, col + dc))
            t = pc.text_of(cell)
            row.append(t)
            row.extend([None] * (cs - 1))
            sp.append((col, cs, t))
            col += cs
        grid.append(row)
        spans.append(sp)
    width = max((len(x) for x in grid), default=0)
    return [x + [None] * (width - len(x)) for x in grid], spans


def col_years(grid, spans):
    """表头带（前 8 行）里的年份 → {col: year_text}
    只在同一单元格的 colspan 范围内分配，不做"向右延续"，
    否则百分比变动列会被错标成左边的年份。"""
    out = {}
    for r in range(min(8, len(grid))):
        row = grid[r]
        years_in_row = [(c, row[c]) for c in range(len(row)) if row[c] and YEARRE.search(row[c])]
        if len(years_in_row) < 2:
            continue
        for start, span, text in spans[r]:
            ys = YEARRE.findall(text or "")
            if not ys:
                continue
            if len(ys) == 1:
                for dc in range(span):
                    out[start + dc] = ys[0]
            else:
                for k, y in enumerate(ys):
                    out[start + k] = y
    return out


def row_years(grid):
    """纵向表：年份写在第一列（"2023 | 1,234"），把年份记到"行"上。
    条件：首列短（≤40 字符）、含年份、且该行其余单元格多数是数字。"""
    out = {}
    for r, row in enumerate(grid):
        first = None
        for c in range(min(2, len(row))):
            if row[c]:
                first = row[c].strip()
                break
        if not first or len(first) > 40:
            continue
        ys = YEARRE.findall(first)
        if len(ys) != 1:
            continue
        # 首列必须是"基本只有年份"的短标签，避免把 "Note 5 (2023 acquisitions)" 当成年份行
        rest = YEARRE.sub("", first)
        rest = re.sub(r"(as of|for the|year|years|ended|december|31,|fiscal|period|and|,|\.|\s)", "", rest, flags=re.I)
        if len(rest) > 12:
            continue
        others = [v for v in row if v and not NUMRE.match(v.strip())]
        nums = [v for v in row if v and NUMRE.match(v.strip())]
        if len(nums) >= 1 and len(nums) >= len(others):
            out[r] = ys[0]
    return out


def row_labels(grid):
    """每行的行标签：左侧第一个含 ≥3 字母、非纯数字的单元格"""
    labels, last = {}, ""
    for r, row in enumerate(grid):
        lab = ""
        for c in range(min(4, len(row))):
            v = row[c]
            if not v:
                continue
            if re.search(r"[A-Za-z]{3,}", v) and not NUMRE.match(v.strip()):
                lab = v.strip()[:120]
            break
        if lab:
            last = lab
        else:
            lab = last          # 多行标签 / 空标签继承上一行
        labels[r] = lab
    return labels


def extract(path):
    doc = os.path.basename(path)[:-5]
    tree = etree.fromstring(open(path, "rb").read(), etree.HTMLParser(recover=True, huge_tree=True))
    for el in [e for e in tree.iter() if isinstance(e.tag, str) and pc.local(e) in ("header", "hidden")]:
        p = el.getparent()
        if p is not None:
            p.remove(el)

    tables, heading = [], ""
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
            junk = bool(JUNK_HEADING.match(t))
            if tag.startswith("h") and len(t) <= 160 and not junk:
                heading = t
            elif not heading and 6 <= len(t) <= 160 and not junk and "table of contents" not in t.lower():
                heading = t
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
            grid, spans = grid_with_spans(trs)
            if not pc.table_ok(grid):
                continue
            tables.append({
                "section": heading,
                "caption": pc.row_text(grid[0])[:200] if grid and any(grid[0]) else "",
                "grid": grid,
                "years": col_years(grid, spans),
                "ryears": row_years(grid),
                "labels": row_labels(grid),
            })
    return doc, tables


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--db", default="finder_tables.db")
    a = ap.parse_args()

    files = sorted(glob.glob("10k/*.html"))[: a.limit or None]
    meta = {r["ticker"]: r for r in csv.DictReader(open(P("10k_manifest.csv"), encoding="utf-8-sig"))}

    for suffix in ("", "-wal", "-shm"):
        if os.path.exists(a.db + suffix):
            os.remove(a.db + suffix)
    con = sqlite3.connect(a.db)
    for stmt in SCHEMA:
        con.execute(stmt)

    t0 = time.time()
    n_tab = n_cell = n_num = n_year = n_label = n_dup = 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, (doc, tables) in enumerate(ex.map(extract, files, chunksize=1), 1):
            m = meta.get(doc, {})
            con.execute("INSERT INTO docs VALUES (?,?,?,?,?,?)",
                        (doc, m.get("ticker", doc), m.get("company", ""), m.get("fiscal_year", ""),
                         m.get("cik", ""), m.get("file", doc + ".html")))
            seen_text = set()          # 同一份 filing 内，完全相同的表只留第一张（脚注重复引用）
            kept_idx = 0
            for src_idx, tb in enumerate(tables, 1):
                text = "\n".join(pc.row_text(r) for r in tb["grid"])
                key = re.sub(r"\s+", " ", text).strip().lower()
                if key in seen_text:
                    n_dup += 1
                    continue
                seen_text.add(key)
                kept_idx += 1
                idx = kept_idx
                grid, years, ryears, labels = tb["grid"], tb["years"], tb["ryears"], tb["labels"]
                crows = []
                for ri, row in enumerate(grid):
                    for ci, v in enumerate(row):
                        if v:
                            yr = years.get(ci) or ryears.get(ri)
                            crows.append((ri, ci, v, to_value(v), yr, labels.get(ri)))
                nums = sum(1 for r in crows if r[3] is not None)
                text = "\n".join(pc.row_text(r) for r in grid)
                cur = con.execute(
                    "INSERT INTO tables(doc_id,table_idx,section,caption,text,n_rows,n_cols,"
                    "n_cells,n_numeric,n_year_cols) VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (doc, idx, tb["section"], tb["caption"], text, len(grid),
                     max((len(r) for r in grid), default=0), len(crows), nums, len(set(years.values()))))
                pk = cur.lastrowid
                con.executemany(
                    "INSERT INTO cells(table_pk,row,col,text,value,year,row_label) VALUES (?,?,?,?,?,?,?)",
                    [(pk, ri, ci, v, val, yr, lb) for ri, ci, v, val, yr, lb in crows])
                con.executemany("INSERT INTO column_map(table_pk,col,year,header_text) VALUES (?,?,?,?)",
                                [(pk, c, y, "") for c, y in sorted(years.items())])
                con.execute("INSERT INTO tables_fts(text,doc_id,table_pk) VALUES (?,?,?)", (text, doc, pk))
                n_tab += 1
                n_cell += len(crows)
                n_num += nums
                n_year += sum(1 for r in crows if r[4])
                n_label += sum(1 for r in crows if r[5])
            if i % 50 == 0 or i == len(files):
                print(f"[{i}/{len(files)}] {time.time()-t0:5.1f}s 表 {n_tab:,} 单元格 {n_cell:,}", flush=True)

    for s in ("CREATE INDEX idx_cells_tp ON cells(table_pk, row, col)",
              "CREATE INDEX idx_cells_val ON cells(value)",
              "CREATE INDEX idx_cells_year ON cells(table_pk, year)",
              "CREATE INDEX idx_tables_doc ON tables(doc_id, table_idx)",
              "CREATE INDEX idx_cmap ON column_map(table_pk, col)"):
        con.execute(s)
    con.commit()
    con.execute("VACUUM")
    con.commit()
    con.close()
    print(f"\n完成 {n_tab:,} 表 / {n_cell:,} 单元格 | 数值 {n_num:,} | "
          f"带年份 {n_year:,} ({100*n_year/max(1,n_cell):.1f}%) | "
          f"带行标签 {n_label:,} ({100*n_label/max(1,n_cell):.1f}%) | "
          f"{time.time()-t0:.1f}s | {os.path.getsize(a.db)/1e6:.0f} MB")


if __name__ == "__main__":
    main()
