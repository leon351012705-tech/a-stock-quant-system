"""
全局配置文件
所有路径、参数、API Key 统一在这里管理
其他模块从这里 import，不要硬编码路径

秘密管理：
  TUSHARE_TOKEN 和 SERVERCHAN_KEY 不在本文件里硬编码。
  优先读环境变量，其次读项目根目录下的 .env 文件（gitignore 排除）。
  .env 格式：每行 KEY=VALUE，# 开头是注释。
"""

import os

# ── 项目根目录（自动定位到本文件所在位置）──
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── 数据库路径 ──
DB_DIR  = os.path.join(BASE_DIR, "db")
DB_PATH = os.path.join(DB_DIR, "market.db")

# ── 日志路径 ──
LOG_DIR = os.path.join(BASE_DIR, "logs")

# ── 数据拉取默认参数 ──
DEFAULT_START_DATE = "20200101"   # 历史数据起始日期
DEFAULT_ADJUST     = "qfq"        # 复权方式：qfq=前复权，hfq=后复权，不填=不复权

# ── 自动创建必要目录 ──
for _dir in [DB_DIR, LOG_DIR]:
    os.makedirs(_dir, exist_ok=True)


# ============================================================
# 秘密加载
# ============================================================

def _load_dotenv(path: str) -> None:
    """简易 .env 解析器（无第三方依赖）。已存在的环境变量不会被覆盖。"""
    if not os.path.isfile(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if k and k not in os.environ:
                os.environ[k] = v


_load_dotenv(os.path.join(BASE_DIR, ".env"))


# ============================================================
# 日志维护
# ============================================================

LOG_RETENTION_DAYS = 30


def cleanup_old_logs(days: int = LOG_RETENTION_DAYS, prefixes: tuple = ("daily_update_", "signal_", "missing_")) -> int:
    """
    删掉 logs/ 下超过 days 天的日志文件。
    只清理匹配 prefixes 之一的文件，避免误删回测结果 PNG/CSV 等。
    返回删除的文件数。
    """
    import time
    if not os.path.isdir(LOG_DIR):
        return 0
    cutoff = time.time() - days * 86400
    removed = 0
    for name in os.listdir(LOG_DIR):
        if not any(name.startswith(p) for p in prefixes):
            continue
        path = os.path.join(LOG_DIR, name)
        try:
            if os.path.isfile(path) and os.path.getmtime(path) < cutoff:
                os.remove(path)
                removed += 1
        except OSError:
            pass
    return removed


# ── Tushare（主通道）- 去 tushare.pro 注册获取 ──
TUSHARE_TOKEN = os.environ.get("TUSHARE_TOKEN", "")

# ── Server酱推送 Key ──
SERVERCHAN_KEY = os.environ.get("SERVERCHAN_KEY", "")

# ── 日常更新策略: "tushare" / "baostock" / "auto" ──
# auto = 先试 tushare，失败自动切 baostock
DAILY_UPDATE_SOURCE = "auto"
