# RAG_FINTECH — 10-K 年报问答 RAG 系统

面向美国上市公司 10-K 年报的检索增强问答系统。核心思路：**不把年报当纯文本切块**，而是按 HTML 原生结构建立章节父子关系，把表格结构化入库并走独立的取数与渲染旁路。

在 FinDER 基准全库 **4,813 道非计算题**上的实测成绩：

| 判分档 | 占比（题数） |
|---|---|
| correct（完全正确） | 17.3%（831） |
| partial（部分正确） | 49.9%（2,403） |
| wrong（事实错误） | 12.6%（608） |
| refused（诚实拒答） | 20.2%（971） |

表特征题（707 道）正确率从 5.4% 提升到 **18.2%**（3.4 倍），拒答从 31.5% 降到 22.3%。

---

## 架构

```
                     ┌──────────────── 离线索引 ────────────────┐
 10-K HTML ──► 结构解析 ──► 父子 chunk ──► BGE-M3 向量 + BM25 ──┐
   (497 份)      │            │                                │
                 │            └─► parent 元数据 (role/年份/邻接) │
                 │                                             │
                 └─► 表格抽取 ──► SQLite cells ──► 表指纹索引 ───┤
                     (含 colspan/年份列还原)   (dense + BM25)   │
                                                               ▼
 问题 ──► doc 路由 ──► ┌ 主路：child dense + BM25 → RRF → rerank ─┐
                       │                                        ├─► 上下文组装 ──► LLM ──► 结构化答案
                       └ 旁路：表指纹召回 → 表渲染(行×年份列) ────┘
```

关键设计：

- **父子 chunk**：parent 承载章节身份与上下文（中位 4,003 字符），child 承载召回（中位 507 字符）；只 embed child，命中后按 `parent_id` 回取整节。
- **表格结构化**：36,393 张表、2.44 M 单元格，按 `colspan` 与表头年份行还原每列年份，取数变成 SQL 而非字符串猜测。
- **表指纹**：`公司 + doc + 章节末两级 + 表名 + 首列前 10 个科目名`，dense + BM25 双路召回 + rerank。
- **表渲染**：不挑行，把命中表整表渲染成"行 × 年份列"交给模型读（表内含金标数字 74%，而单行 fact 只有 19%）。

---

## 目录

```
RAG_FINTECH/
├── config.py                 # 数据目录、凭证、检索超参（唯一配置入口）
├── .env.example
├── requirements.txt
├── src/
│   ├── parse/chunker.py           # 结构感知父子切分（HTML → parent/child）
│   ├── index/
│   │   ├── embed_children.py      # child 向量化（BGE-M3, fp16, 可续跑）
│   │   ├── build_bm25.py          # child 稀疏索引
│   │   ├── build_doc_index.py     # child → doc 索引（doc 内检索）
│   │   ├── enrich_parent_meta.py  # parent 元数据（role/年份/邻接）
│   │   ├── build_table_sqlite.py  # 表格 → SQLite（含年份列还原）
│   │   └── build_table_index.py   # 表指纹 → 向量 + BM25
│   ├── rag/
│   │   └── retriever.py           # 路由 / 双路召回 / RRF / rerank / 表旁路
│   │                              # 含上下文组装 build_ctx() 与作答入口 answer()/ask()
│   ├── generate/prompts.py        # v9 作答 prompt + 判分 prompt（唯一真源）
│   └── eval/
│       ├── build_qrels.py         # 由 references 投影生成检索 qrels
│       ├── hit_at_k.py            # hit@k / MRR 评测
│       └── judge.py               # 答案级 LLM 判分
├── scripts/
│   ├── run_full.py                # 全库批跑：ctx → gen → judge → report（可续跑）
│   ├── render_tables_707.py       # 表渲染旁路批跑
│   ├── eval_table_ab.py           # 表路径 A/B 对照
│   └── smoke_test.py              # 单题冒烟测试
├── docs/FINDINGS.md               # 全部实验结果与负面结论
└── data/README.md                 # 数据放置说明（大文件不入库）
```

---

## 快速开始

```bash
# 1. 依赖
pip install -r requirements.txt

# 2. 配置（复制后填 API Key；也可用环境变量覆盖）
cp .env.example .env

# 3. 指向数据目录（默认 D:/RAG/data/finder）
export RAG_DATA_DIR=/path/to/finder_data

# 4. 冒烟测试：单题跑通检索 + 作答
python scripts/smoke_test.py "What was Lowe's cash dividend payments in 2023?"
```

## 完整流水线

按顺序执行（每一步都可中断续跑）：

```bash
# ① 结构解析 + 父子切分 → chunks_struct.jsonl
python src/parse/chunker.py --workers 10

# ② 元数据与索引
python src/index/enrich_parent_meta.py
python src/index/build_doc_index.py
python src/index/embed_children.py          # BGE-M3, 551,692 条
python src/index/build_bm25.py

# ③ 表格
python src/index/build_table_sqlite.py      # → finder_tables.db
python src/index/build_table_index.py       # → 表指纹向量 + BM25

# ④ 评测基线
python src/eval/build_qrels.py              # → qrels（由 references 投影）
python src/eval/hit_at_k.py --n 600

# ⑤ 全库跑：上下文 → 作答 → 判分 → 报告
python scripts/run_full.py --phase ctx
python scripts/run_full.py --phase gen      # 8 线程
python scripts/run_full.py --phase judge
python scripts/run_full.py --phase report
```

若表题走了改进版旁路，先合并再报告：

```bash
python scripts/render_tables_707.py            # 表渲染上下文
python scripts/eval_table_ab.py --mode dense   # 作答 + 判分
python scripts/merge_final.py                  # → verdicts_final.jsonl
python scripts/run_full.py --phase report      # 优先读 verdicts_final
```


---

## 超参

集中在 `config.py`，不必翻脚本：

| 参数 | 默认 | 说明 |
|---|---|---|
| `N_PARENTS` | 5 | 送入上下文的父段落数 |
| `CAND_K` | 50 | child 候选池 |
| `SIDE_TOPK` | 5 | 表旁路候选表数 |
| `N_EVID` | 3 | 每段落证据句数 |
| `CHILD_MAX_CHARS` | 14000 | 上下文上限 |

对照实验可用环境变量 `FR_LOCATE2=0` 关闭新版行匹配（反向覆盖 + IDF），退回旧版（严格门槛）。

---

## 已知边界

- 计算题（883 道，需多步算术）未纳入本轮评测；
- 表特征题仍有 22.3% 拒答，未逐类归因；
- `Governance`（7%）与 `Financials`（8%）两类明显低于其他类别，多为跨节组合题；
- 检索侧 hit@5 = 51%，被 qrels 噪声低估（详见 `docs/FINDINGS.md`），已停止在该方向继续投入。

## 数据

497 份 10-K（2.14 GB）、向量索引（1.13 GB）等大文件不纳入仓库，见 `data/README.md`。
