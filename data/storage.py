"""
【数据层】storage.py — SQLite 统一存储接口

职责：
  - 管理数据库连接和表结构
  - 提供写入、查询的标准接口
  - 其他层只调用这里的函数，不直接操作数据库

设计原则：
  - 所有接口返回 pandas DataFrame，方便研究层/信号层直接使用
  - 写入时自动去重（按股票代码+日期唯一）
  - 出错时记录日志而不是直接崩溃
"""

import sqlite3
import pandas as pd
import logging
import os
import sys

# 把项目根目录加入路径，确保能 import config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DB_PATH

logger = logging.getLogger(__name__)


def get_connection() -> sqlite3.Connection:
    """
    获取数据库连接。
    每次调用返回新连接，使用完毕后请关闭（或用 with 语句）。

    PRAGMA 配置（每个连接生效，WAL 是数据库级持久设置）：
      - journal_mode=WAL    : 写不阻塞读，比默认 DELETE 模式快 2-5x
      - synchronous=NORMAL  : 配合 WAL 使用是安全的，比 FULL 快很多
      - cache_size=-65536   : 64MB 页缓存（默认仅 2MB），对 1GB+ 数据库帮助显著
      - temp_store=MEMORY   : 临时表/索引放内存
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA cache_size=-65536")
    conn.execute("PRAGMA temp_store=MEMORY")
    return conn


def init_database():
    """
    初始化数据库，创建所有必要的表。
    可重复调用（已存在的表不会重建）。
    """
    conn = get_connection()
    try:
        cursor = conn.cursor()

        # ── 日线行情表 ──
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS daily_bars (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol      TEXT    NOT NULL,
                trade_date  TEXT    NOT NULL,
                open        REAL,
                high        REAL,
                low         REAL,
                close       REAL,
                volume      REAL,
                amount      REAL,
                amplitude   REAL,
                pct_change  REAL,
                change      REAL,
                turnover    REAL,
                updated_at  TEXT DEFAULT (datetime('now', 'localtime')),
                UNIQUE(symbol, trade_date)
            )
        """)

        # ── 股票基本信息表 ──
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS stock_info (
                symbol      TEXT PRIMARY KEY,
                name        TEXT,
                industry    TEXT,
                market      TEXT,
                updated_at  TEXT DEFAULT (datetime('now', 'localtime'))
            )
        """)

        # ── 索引：按 trade_date 单独筛选用（市场广度、单日全市场扫描）──
        # UNIQUE(symbol, trade_date) 自带的索引以 symbol 为前缀，无法加速纯日期过滤
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_daily_bars_date
            ON daily_bars(trade_date)
        """)

        # ── 交易日历表（A 股法定节假日跳过用）──
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trade_calendar (
                cal_date    TEXT PRIMARY KEY,    -- 'YYYY-MM-DD'
                is_open     INTEGER NOT NULL     -- 1=开市, 0=休市
            )
        """)

        conn.commit()
        logger.info("数据库初始化完成：%s", DB_PATH)

    except Exception as e:
        logger.error("数据库初始化失败：%s", e)
        raise
    finally:
        conn.close()


# ════════════════════════════════════════
#  写入接口
# ════════════════════════════════════════

def save_daily_bars(df: pd.DataFrame, symbol: str) -> int:
    """
    将日线数据写入数据库。
    重复数据（同一股票+同一日期）自动跳过，不报错。

    参数：
        df      : akshare 返回的日线 DataFrame
        symbol  : 股票代码，如 '000001'

    返回：
        实际新增写入的记录数量
    """
    if df is None or df.empty:
        logger.warning("[%s] 传入数据为空，跳过写入", symbol)
        return 0

    # ── 标准化列名（akshare 返回中文列名，统一转成英文）──
    column_map = {
        "日期":   "trade_date",
        "开盘":   "open",
        "最高":   "high",
        "最低":   "low",
        "收盘":   "close",
        "成交量": "volume",
        "成交额": "amount",
        "振幅":   "amplitude",
        "涨跌幅": "pct_change",
        "涨跌额": "change",
        "换手率": "turnover",
    }
    df = df.rename(columns=column_map)

    # ── 确保有 trade_date 列 ──
    if "trade_date" not in df.columns:
        logger.error("[%s] 数据缺少日期列，写入失败", symbol)
        return 0

    # ── 统一日期格式为 YYYY-MM-DD ──
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.strftime("%Y-%m-%d")
    df["symbol"] = symbol

    # ── 只保留数据库表中有的列 ──
    keep_cols = [
        "symbol", "trade_date", "open", "high", "low", "close",
        "volume", "amount", "amplitude", "pct_change", "change", "turnover"
    ]
    df = df[[c for c in keep_cols if c in df.columns]]

    conn = get_connection()
    try:
        inserted_count = df.to_sql(
            name="daily_bars",
            con=conn,
            if_exists="append",
            index=False,
            method=_insert_or_ignore,
        )

        if inserted_count is None:
            inserted_count = 0

        conn.commit()
        logger.info("[%s] 写入完成：输入 %d 条，实际新增 %d 条", symbol, len(df), inserted_count)
        return int(inserted_count)

    except Exception as e:
        logger.error("[%s] 写入失败：%s", symbol, e)
        conn.rollback()
        return 0
    finally:
        conn.close()


def _insert_or_ignore(table, conn, keys, data_iter):
    """
    pandas.to_sql 的自定义插入方法：
    使用 INSERT OR IGNORE 避免重复报错，并返回实际插入条数。

    用 executemany 一次提交所有行，比逐行 execute 快 5-10x。
    """
    cols = ", ".join(keys)
    placeholders = ", ".join(["?" for _ in keys])
    sql = f"INSERT OR IGNORE INTO {table.name} ({cols}) VALUES ({placeholders})"

    rows = list(data_iter)
    if not rows:
        return 0

    cursor = conn.execute("BEGIN")
    cursor = conn.executemany(sql, rows)
    return cursor.rowcount


# ════════════════════════════════════════
#  查询接口
# ════════════════════════════════════════

def query_daily_bars(
    symbol: str,
    start_date: str = None,
    end_date: str = None,
) -> pd.DataFrame:
    """
    查询某只股票的日线数据。

    参数：
        symbol     : 股票代码，如 '000001'
        start_date : 开始日期，如 '2024-01-01'（可选）
        end_date   : 结束日期，如 '2024-12-31'（可选）

    返回：
        DataFrame，按日期升序排列，列名为英文
    """
    conditions = ["symbol = ?"]
    params = [symbol]

    if start_date:
        conditions.append("trade_date >= ?")
        params.append(start_date)
    if end_date:
        conditions.append("trade_date <= ?")
        params.append(end_date)

    where_clause = " AND ".join(conditions)
    sql = f"SELECT * FROM daily_bars WHERE {where_clause} ORDER BY trade_date ASC"

    conn = get_connection()
    try:
        df = pd.read_sql_query(sql, conn, params=params)
        return df
    except Exception as e:
        logger.error("查询 [%s] 失败：%s", symbol, e)
        return pd.DataFrame()
    finally:
        conn.close()


def get_latest_date(symbol: str) -> str | None:
    """
    查询某只股票在数据库中最新的交易日期。
    用于增量更新：只拉取数据库没有的部分。

    返回：
        最新日期字符串如 '2024-03-20'，若无数据则返回 None
    """
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT MAX(trade_date) FROM daily_bars WHERE symbol = ?",
            (symbol,)
        )
        result = cursor.fetchone()[0]
        return result
    except Exception as e:
        logger.error("查询最新日期失败 [%s]：%s", symbol, e)
        return None
    finally:
        conn.close()


def get_data_summary() -> pd.DataFrame:
    """
    查看数据库中各股票的数据概况。

    返回：
        DataFrame，包含 symbol, record_count, earliest_date, latest_date
    """
    sql = """
        SELECT
            symbol,
            COUNT(*)        AS record_count,
            MIN(trade_date) AS earliest_date,
            MAX(trade_date) AS latest_date
        FROM daily_bars
        GROUP BY symbol
        ORDER BY symbol
    """
    conn = get_connection()
    try:
        return pd.read_sql_query(sql, conn)
    except Exception as e:
        logger.error("查询数据概况失败：%s", e)
        return pd.DataFrame()
    finally:
        conn.close()


def get_database_latest_trade_date() -> str | None:
    """
    查询数据库中全市场的最新交易日。

    返回：
        YYYY-MM-DD 或 None
    """
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT MAX(trade_date) FROM daily_bars")
        result = cursor.fetchone()[0]
        return result
    except Exception as e:
        logger.error("查询数据库最新交易日失败：%s", e)
        return None
    finally:
        conn.close()


def get_symbols_need_update(target_trade_date: str) -> list[str]:
    """
    查询哪些股票尚未更新到目标交易日。

    参数：
        target_trade_date : 目标交易日，格式 YYYY-MM-DD

    返回：
        需要更新的股票代码列表
    """
    sql = """
        SELECT s.symbol
        FROM stock_info s
        LEFT JOIN (
            SELECT symbol, MAX(trade_date) AS latest_date
            FROM daily_bars
            GROUP BY symbol
        ) d
        ON s.symbol = d.symbol
        WHERE d.latest_date IS NULL OR d.latest_date < ?
        ORDER BY s.symbol
    """
    conn = get_connection()
    try:
        df = pd.read_sql_query(sql, conn, params=(target_trade_date,))
        return df["symbol"].astype(str).tolist() if not df.empty else []
    except Exception as e:
        logger.error("查询待更新股票失败：%s", e)
        return []
    finally:
        conn.close()


def save_daily_snapshot(df: pd.DataFrame) -> int:
    """
    将全市场单日快照写入 daily_bars。
    说明：
      - 仅用于日常盘后更新“当天这一根K线”
      - 不做复权重算，直接把当天快照入库
      - 若 (symbol, trade_date) 已存在，则忽略

    期望输入列（东方财富/akshare常见中文列名）：
      代码, 最新价/收盘, 今开, 最高, 最低, 成交量, 成交额, 涨跌幅, 涨跌额, 换手率
    """
    if df is None or df.empty:
        logger.warning("快照数据为空，跳过写入")
        return 0

    raw = df.copy()

    column_candidates = {
        "symbol": ["代码", "symbol"],
        "close": ["最新价", "收盘", "close"],
        "open": ["今开", "开盘", "open"],
        "high": ["最高", "high"],
        "low": ["最低", "low"],
        "volume": ["成交量", "volume"],
        "amount": ["成交额", "amount"],
        "pct_change": ["涨跌幅", "pct_change"],
        "change": ["涨跌额", "change"],
        "turnover": ["换手率", "turnover"],
    }

    mapped = {}
    for target, candidates in column_candidates.items():
        for c in candidates:
            if c in raw.columns:
                mapped[target] = c
                break

    required = ["symbol", "close", "open", "high", "low", "volume", "amount", "pct_change"]
    missing = [c for c in required if c not in mapped]
    if missing:
        logger.error("快照数据缺少必要列映射：%s | 实际列名：%s", missing, list(raw.columns))
        return 0

    today_str = pd.Timestamp.today().strftime("%Y-%m-%d")

    out = pd.DataFrame()
    out["symbol"] = raw[mapped["symbol"]].astype(str).str.zfill(6)
    out["trade_date"] = today_str
    out["open"] = pd.to_numeric(raw[mapped["open"]], errors="coerce")
    out["high"] = pd.to_numeric(raw[mapped["high"]], errors="coerce")
    out["low"] = pd.to_numeric(raw[mapped["low"]], errors="coerce")
    out["close"] = pd.to_numeric(raw[mapped["close"]], errors="coerce")
    out["volume"] = pd.to_numeric(raw[mapped["volume"]], errors="coerce")
    out["amount"] = pd.to_numeric(raw[mapped["amount"]], errors="coerce")
    out["pct_change"] = pd.to_numeric(raw[mapped["pct_change"]], errors="coerce")

    out["change"] = pd.to_numeric(raw[mapped["change"]], errors="coerce") if "change" in mapped else None
    out["turnover"] = pd.to_numeric(raw[mapped["turnover"]], errors="coerce") if "turnover" in mapped else None
    out["amplitude"] = None

    out = out.dropna(subset=["symbol", "trade_date", "open", "high", "low", "close"])
    out = out[[
        "symbol", "trade_date", "open", "high", "low", "close",
        "volume", "amount", "amplitude", "pct_change", "change", "turnover"
    ]]

    if out.empty:
        logger.warning("快照数据清洗后为空，跳过写入")
        return 0

    conn = get_connection()
    try:
        inserted_count = out.to_sql(
            name="daily_bars",
            con=conn,
            if_exists="append",
            index=False,
            method=_insert_or_ignore,
        )
        if inserted_count is None:
            inserted_count = 0
        conn.commit()
        logger.info("快照写入完成：输入 %d 条，实际新增 %d 条", len(out), inserted_count)
        return int(inserted_count)
    except Exception as e:
        logger.error("快照写入失败：%s", e)
        conn.rollback()
        return 0
    finally:
        conn.close()
# ════════════════════════════════════════
#  批量写入接口（供 fetcher_daily.py 使用）
# ════════════════════════════════════════

def save_daily_bars_bulk(df: pd.DataFrame) -> int:
    """
    将全市场日K数据批量写入 daily_bars。
    专为 fetcher_daily.py（Tushare / BaoStock 通道）设计。

    期望输入列:
        code, date, open, high, low, close, volume, amount
        可选: pct_change, change, turnover

    与现有 save_daily_bars(df, symbol) 的区别:
        - 这个函数接受多只股票的合并DataFrame
        - 列名是英文（不需要中文映射）
        - 缺失的列自动填 None

    返回:
        实际新增写入的记录数量
    """
    if df is None or df.empty:
        logger.warning("批量写入：传入数据为空，跳过")
        return 0

    out = pd.DataFrame()

    # ── 必要列映射 ──
    out["symbol"] = df["code"].astype(str).str.zfill(6)
    out["trade_date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    out["open"] = pd.to_numeric(df["open"], errors="coerce")
    out["high"] = pd.to_numeric(df["high"], errors="coerce")
    out["low"] = pd.to_numeric(df["low"], errors="coerce")
    out["close"] = pd.to_numeric(df["close"], errors="coerce")
    out["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    out["amount"] = pd.to_numeric(df["amount"], errors="coerce")

    # ── 可选列（有就用，没有就填 None）──
    out["amplitude"] = pd.to_numeric(df["amplitude"], errors="coerce") \
        if "amplitude" in df.columns else None
    out["pct_change"] = pd.to_numeric(df["pct_change"], errors="coerce") \
        if "pct_change" in df.columns else None
    out["change"] = pd.to_numeric(df["change"], errors="coerce") \
        if "change" in df.columns else None
    out["turnover"] = pd.to_numeric(df["turnover"], errors="coerce") \
        if "turnover" in df.columns else None

    # ── 去掉关键列为空的行 ──
    out = out.dropna(subset=["symbol", "trade_date", "open", "high", "low", "close"])

    # ── 保证列顺序与表结构一致 ──
    out = out[[
        "symbol", "trade_date", "open", "high", "low", "close",
        "volume", "amount", "amplitude", "pct_change", "change", "turnover"
    ]]

    if out.empty:
        logger.warning("批量写入：清洗后为空，跳过")
        return 0

    conn = get_connection()
    try:
        inserted_count = out.to_sql(
            name="daily_bars",
            con=conn,
            if_exists="append",
            index=False,
            method=_insert_or_ignore,
        )
        if inserted_count is None:
            inserted_count = 0
        conn.commit()
        logger.info(
            "批量写入完成：输入 %d 条，实际新增 %d 条，覆盖 %d 只股票",
            len(out), inserted_count, out["symbol"].nunique()
        )
        return int(inserted_count)
    except Exception as e:
        logger.error("批量写入失败：%s", e)
        conn.rollback()
        return 0
    finally:
        conn.close()