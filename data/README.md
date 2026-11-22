# 数据放置说明

代码仓库只保留脚本，数据与索引都放在 `DATA_DIR`（默认 `D:/RAG/data/finder`，可用环境变量 `RAG_DATA_DIR` 覆盖）。

## 目录结构

```
DATA_DIR/
├── 10k/                       # 497 份 10-K HTML（2.14 GB）
├── 10k_manifest.csv           # ticker / 公司 / 财年 / CIK / 文件大小
├── references.jsonl           # 5,703 题：query / answer / references
│
├── chunks_struct.jsonl        # 父子 chunk（656,808 条，约 615 MB）
├── parent_meta.jsonl          # parent 元数据（role / 年份 / prev-next）
├── chunk_ids.jsonl            # 向量行号 ↔ chunk_id
├── child_doc_idx.npy          # child → doc 编号
├── child_docs.json            # doc 编号 → ticker
│
├── vectors_dense.npy          # child 向量（551,692 × 1024, fp16, 1.13 GB）
├── bm25.npz / bm25_vocab.json # child 稀疏索引
│
├── finder_tables.db           # 表格 SQLite（36,393 表 / 2.44 M 单元格，约 216 MB）
├── tables_db.jsonl            # 表指纹
├── tables_db_vec.npy          # 表指纹向量
├── tables_db_bm25.npz / tables_db_bm25_vocab.json
│
├── qrels_text_v2.json         # 检索 qrels（由 references 投影）
└── ctx_all.jsonl / answers_full.jsonl / verdicts_full.jsonl   # 批跑产物
```

## 获取源数据

FinDER 是公开基准（LinqAlpha），数据入口：

- 数据集：`https://huggingface.co/datasets/Linq-AI-Research/FinDER`
- 论文：`https://arxiv.org/abs/2504.15800`

解压后把 `10-k.zip` 展开到 `10k/`，即可按 README 的流水线重新生成其余产物。

## 空间占用

| 部分 | 大小 |
|---|---|
| 原始 HTML | 2.14 GB |
| chunk 文本 | 615 MB |
| child 向量 | 1.13 GB |
| 表格 SQLite | 216 MB |
| 其余索引 | < 200 MB |

合计约 4.3 GB。建议全流程在 SSD 上跑；向量与 SQLite 均支持 mmap / 只读并发。
