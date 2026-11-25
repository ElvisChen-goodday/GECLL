"""全量向量化：551,692 个 child → vectors_dense.npy (fp16) + chunk_ids.jsonl

- 固定顺序：按 chunks_struct.jsonl 中出现顺序，ids[i] ↔ vectors[i]
- 分块编码（每 4,000 条一次 encode 调用），边编码边写入预分配 memmap，崩溃不丢已算部分
- 每 20,000 条打印进度与 ETA
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
from config import P, DATA_DIR, ENV_FILE, load_env
import os, json, time, argparse
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_OFFLINE", os.environ.get("RAG_OFFLINE", "1"))

import numpy as np
import torch
from FlagEmbedding import BGEM3FlagModel


SRC = "chunks_struct.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--slice", type=int, default=4000)
    ap.add_argument("--out", default="vectors_dense.npy")
    ap.add_argument("--ids", default="chunk_ids.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    ids, texts = [], []
    with open(SRC, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["level"] != "child":
                continue
            ids.append(r["chunk_id"])
            texts.append(r["text"])
            if a.limit and len(ids) >= a.limit:
                break
    n = len(ids)
    print(f"待编码 child: {n:,}", flush=True)

    t0 = time.time()
    model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True, device=os.environ.get("RAG_DEVICE", "cuda"))
    print(f"模型就绪 {time.time()-t0:.1f}s", flush=True)

    vecs = np.lib.format.open_memmap(a.out, mode="w+", dtype=np.float16, shape=(n, 1024))
    t1 = time.time()
    done = 0
    for i in range(0, n, a.slice):
        chunk = texts[i:i + a.slice]
        out = model.encode(chunk, batch_size=a.batch, max_length=a.max_len)["dense_vecs"]
        vecs[i:i + len(chunk)] = np.asarray(out, dtype=np.float16)
        vecs.flush()
        done = i + len(chunk)
        if done % 20000 < a.slice or done == n:
            el = time.time() - t1
            rate = done / el
            print(f"[{done:,}/{n:,}] {el:6.1f}s | {rate:6.0f} chunks/s | "
                  f"ETA {(n-done)/max(rate,1e-9)/60:4.1f} min | "
                  f"VRAM {torch.cuda.max_memory_allocated()/1e9:.2f} GB", flush=True)

    with open(a.ids, "w", encoding="utf-8") as f:
        for cid in ids:
            f.write(json.dumps({"chunk_id": cid}, ensure_ascii=False) + "\n")

    # 自校验
    arr = np.load(a.out, mmap_mode="r")
    norms = np.linalg.norm(arr[:200].astype(np.float32), axis=1)
    elapsed = time.time() - t1
    print(f"\n完成: {n:,} 条 / {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"vectors {arr.shape} {arr.dtype} | {os.path.getsize(a.out)/1e9:.2f} GB")
    print(f"ids 行数 {sum(1 for _ in open(a.ids, encoding='utf-8')):,}")
    print(f"前 200 条向量 L2 范数: 均值 {norms.mean():.3f} 最小 {norms.min():.3f} 最大 {norms.max():.3f}")
    print(f"行数对齐: {arr.shape[0] == n}")


if __name__ == "__main__":
    main()
