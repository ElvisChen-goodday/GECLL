"""全局配置：数据目录、凭证、检索超参

解析优先级（前一个命中即用）：
  1. 环境变量        RAG_DATA_DIR / RAG_ENV_FILE
  2. 仓库内 .env     （可写 RAG_DATA_DIR / RAG_ENV_FILE，不含密钥，可提交）
  3. 默认值          DATA_DIR = <repo>/data，ENV_FILE = <repo>/.env

密钥只放在 ENV_FILE 指向的文件里（默认 <repo>/.env，已被 .gitignore 忽略）。
"""
import os

# ---- 代码根目录 ----
ROOT = os.path.dirname(os.path.abspath(__file__))
_PKG_ENV = os.path.join(ROOT, ".env")


def _read_env_file(path):
    out = {}
    if path and os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip()
    return out


_pkg = _read_env_file(_PKG_ENV)

# ---- 数据根目录：大文件不入库，放在这里 ----
DATA_DIR = (os.environ.get("RAG_DATA_DIR")
            or _pkg.get("RAG_DATA_DIR")
            or os.path.join(ROOT, "data"))

# ---- 凭证文件 ----
ENV_FILE = (os.environ.get("RAG_ENV_FILE")
            or _pkg.get("RAG_ENV_FILE")
            or _PKG_ENV)


def P(name: str) -> str:
    """把产物文件名解析成 DATA_DIR 下的绝对路径"""
    if os.path.isabs(name):
        return name
    return os.path.join(DATA_DIR, name)


def load_env(path: str = None) -> dict:
    """读取 KEY=VALUE 形式的 .env（不覆盖已存在的环境变量）"""
    path = path or ENV_FILE
    env = {}
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    for k, v in env.items():
        os.environ.setdefault(k, v)
    return env


# ---- 检索/生成超参（改这里，不要散落在脚本里）----
N_PARENTS = 5        # 送入上下文的父段落数
CAND_K = 50          # child 候选池大小
SIDE_TOPK = 5        # 表旁路候选表数
N_EVID = 3           # 每个父段落送几条证据句
CHILD_MAX_CHARS = 14000   # 上下文长度上限
