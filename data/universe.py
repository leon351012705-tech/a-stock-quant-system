"""
【数据层】universe.py — 股票池管理模块

职责：
  - 从 akshare 拉取全 A 股股票列表（代码 + 名称 + 所属市场）
  - 将列表存入 stock_info 表
  - 提供"我现在要扫描哪些股票"的查询接口

使用方式：
  from data.universe import update_stock_list, get_all_symbols
  update_stock_list()       # 更新一次全市场列表（每周跑一次即可）
  symbols = get_all_symbols()  # 拿到所有股票代码列表
"""

import akshare as ak
import pandas as pd
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.storage import get_connection

logger = logging.getLogger(__name__)


def update_stock_list() -> int:
    """
    从 akshare 拉取全 A 股列表，更新 stock_info 表。
    已存在的股票会更新名称和市场信息，新股自动插入。

    返回：
        写入/更新的股票数量
    """
    logger.info("正在拉取全 A 股列表...")
    try:
        # stock_info_a_code_name() 返回所有A股的代码和名称
        df = ak.stock_info_a_code_name()
    except Exception as e:
        logger.error("拉取股票列表失败：%s", e)
        return 0

    if df is None or df.empty:
        logger.warning("股票列表为空")
        return 0

    # ── 标准化列名 ──
    df = df.rename(columns={"code": "symbol", "name": "name"})

    # ── 根据代码前缀判断市场 ──
    def infer_market(code: str) -> str:
        if code.startswith("6"):
            return "SH"   # 上交所
        elif code.startswith(("0", "3")):
            return "SZ"   # 深交所
        elif code.startswith(("4", "8", "9")):
            return "BJ"   # 北交所
        return "未知"

    df["market"] = df["symbol"].apply(infer_market)
    df["industry"] = ""   # 行业数据后续单独补充

    # ── 写入数据库，已存在则更新（executemany 批量提交，比逐行快 5-10x）──
    rows = list(zip(
        df["symbol"].tolist(),
        df["name"].tolist(),
        df["market"].tolist(),
        df["industry"].tolist(),
    ))

    conn = get_connection()
    try:
        conn.executemany("""
            INSERT INTO stock_info (symbol, name, market, industry, updated_at)
            VALUES (?, ?, ?, ?, datetime('now', 'localtime'))
            ON CONFLICT(symbol) DO UPDATE SET
                name       = excluded.name,
                market     = excluded.market,
                updated_at = excluded.updated_at
        """, rows)
        conn.commit()
        count = len(rows)
        logger.info("股票列表更新完成，共 %d 只", count)
        return count
    except Exception as e:
        logger.error("写入股票列表失败：%s", e)
        conn.rollback()
        return 0
    finally:
        conn.close()


def get_all_symbols(market: str = None) -> list:
    """
    从数据库获取股票代码列表。

    参数：
        market : 可选过滤，'SH' / 'SZ' / 'BJ'，不填则返回全市场

    返回：
        股票代码字符串列表，如 ['000001', '000002', ...]
    """
    conn = get_connection()
    try:
        if market:
            df = pd.read_sql_query(
                "SELECT symbol FROM stock_info WHERE market = ? ORDER BY symbol",
                conn, params=(market,)
            )
        else:
            df = pd.read_sql_query(
                "SELECT symbol FROM stock_info ORDER BY symbol",
                conn
            )
        return df["symbol"].tolist()
    except Exception as e:
        logger.error("获取股票列表失败：%s", e)
        return []
    finally:
        conn.close()


def get_stock_name(symbol: str) -> str:
    """
    查询单只股票的名称。

    返回：
        股票名称，查不到则返回空字符串
    """
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM stock_info WHERE symbol = ?", (symbol,))
        row = cursor.fetchone()
        return row["name"] if row else ""
    except Exception as e:
        logger.error("查询股票名称失败 [%s]：%s", symbol, e)
        return ""
    finally:
        conn.close()


def get_universe_summary() -> dict:
    """
    返回股票池概况：各市场数量。
    """
    conn = get_connection()
    try:
        df = pd.read_sql_query(
            "SELECT market, COUNT(*) AS cnt FROM stock_info GROUP BY market",
            conn
        )
        return dict(zip(df["market"], df["cnt"]))
    except Exception as e:
        logger.error("获取股票池概况失败：%s", e)
        return {}
    finally:
        conn.close()
