"""
data/trade_calendar.py — A 股交易日历（识别法定节假日）

数据源优先级：
  1. Tushare trade_cal（需 120 积分权限）
  2. akshare tool_trade_date_hist_sina（免费兜底）

存入 daily_bars 数据库的 trade_calendar 表，每年首次启动拉一次即可。

提供查询：
  is_trading_day(d)                   -- 该日是否开市
  next_trading_day(after_date)        -- after_date 之后第一个开市日
  latest_trading_day_on_or_before(d)  -- d 当天或之前最近一个开市日
"""

from __future__ import annotations
import logging
from datetime import date, timedelta
from data.storage import get_connection

logger = logging.getLogger(__name__)


def _write_rows(rows: list, source: str) -> int:
    """共用的写入逻辑"""
    if not rows:
        return 0
    conn = get_connection()
    try:
        conn.executemany(
            "INSERT OR REPLACE INTO trade_calendar (cal_date, is_open) VALUES (?, ?)",
            rows,
        )
        conn.commit()
        logger.info("[trade_calendar] 已写入 %d 条（来源: %s, %s ~ %s）",
                    len(rows), source, rows[0][0], rows[-1][0])
        return len(rows)
    except Exception as e:
        logger.error("[trade_calendar] 写入失败：%s", e)
        conn.rollback()
        return 0
    finally:
        conn.close()


def _via_tushare(token: str, start: str, end: str) -> int:
    """Tushare 通道。失败返回 0，让上层 fall back。"""
    try:
        import tushare as ts
        pro = ts.pro_api(token)
        # Tushare 用 YYYYMMDD 格式
        ts_start = start.replace("-", "")
        ts_end = end.replace("-", "")
        df = pro.trade_cal(start_date=ts_start, end_date=ts_end)
    except Exception as e:
        # 权限错误等，静悄悄返回 0 让 akshare 接力
        logger.info("[trade_calendar] Tushare 不可用（%s），切 akshare", str(e)[:80])
        return 0

    if df is None or df.empty:
        return 0

    rows = []
    for _, r in df.iterrows():
        cal = str(r["cal_date"])
        cal_iso = f"{cal[:4]}-{cal[4:6]}-{cal[6:8]}"
        rows.append((cal_iso, int(r["is_open"])))

    return _write_rows(rows, "Tushare")


def _via_akshare(start: str, end: str) -> int:
    """akshare 通道。返回历史所有交易日，自己生成完整日期+is_open 标志。"""
    try:
        import akshare as ak
        # tool_trade_date_hist_sina() 返回所有历史交易日（无参数）
        df = ak.tool_trade_date_hist_sina()
    except Exception as e:
        logger.error("[trade_calendar] akshare 拉取失败：%s", e)
        return 0

    if df is None or df.empty:
        logger.warning("[trade_calendar] akshare 返回空")
        return 0

    # 提取所有交易日为 set
    trading_days = set()
    col = "trade_date" if "trade_date" in df.columns else df.columns[0]
    for v in df[col]:
        if hasattr(v, "strftime"):
            trading_days.add(v.strftime("%Y-%m-%d"))
        else:
            s = str(v).replace("/", "-")
            # 标准化 'YYYY-MM-DD'
            if len(s) >= 10:
                trading_days.add(s[:10])

    # 在 [start, end] 范围内生成完整日期序列，标 is_open
    start_d = date.fromisoformat(start)
    end_d = date.fromisoformat(end)
    rows = []
    d = start_d
    while d <= end_d:
        ds = d.strftime("%Y-%m-%d")
        is_open = 1 if ds in trading_days else 0
        rows.append((ds, is_open))
        d += timedelta(days=1)

    return _write_rows(rows, "akshare")


def update_trade_calendar(token: str = "",
                          start: str = "2020-01-01",
                          end: str = "2030-12-31") -> int:
    """
    更新交易日历。优先 Tushare，失败兜 akshare。
    """
    # 1) 试 Tushare
    if token:
        n = _via_tushare(token, start, end)
        if n > 0:
            return n

    # 2) akshare 兜底
    return _via_akshare(start, end)


def calendar_count() -> int:
    """日历表里的记录数（用于判断是否需要 update）"""
    try:
        conn = get_connection()
        n = conn.execute("SELECT COUNT(*) FROM trade_calendar").fetchone()[0]
        conn.close()
        return int(n)
    except Exception:
        return 0


def latest_trading_day_on_or_before(d: str) -> str | None:
    """返回 <= d 的最近一个开市日。若日历为空返回 None。"""
    try:
        conn = get_connection()
        row = conn.execute(
            "SELECT cal_date FROM trade_calendar "
            "WHERE cal_date <= ? AND is_open = 1 "
            "ORDER BY cal_date DESC LIMIT 1",
            (d,),
        ).fetchone()
        conn.close()
        return row[0] if row else None
    except Exception:
        return None


def next_trading_day(after_date: str) -> str | None:
    """返回 > after_date 的第一个开市日。若日历为空返回 None。"""
    try:
        conn = get_connection()
        row = conn.execute(
            "SELECT cal_date FROM trade_calendar "
            "WHERE cal_date > ? AND is_open = 1 "
            "ORDER BY cal_date ASC LIMIT 1",
            (after_date,),
        ).fetchone()
        conn.close()
        return row[0] if row else None
    except Exception:
        return None


def is_trading_day(d: str) -> bool | None:
    """
    返回 d 是否开市。
    True/False 是日历给的答案；None 表示日历里查不到（无数据，外部应回退到默认逻辑）。
    """
    try:
        conn = get_connection()
        row = conn.execute(
            "SELECT is_open FROM trade_calendar WHERE cal_date = ?",
            (d,),
        ).fetchone()
        conn.close()
        if row is None:
            return None
        return bool(row[0])
    except Exception:
        return None


def _calendar_max_date() -> str | None:
    """返回日历表里最大日期，空表返回 None"""
    try:
        conn = get_connection()
        row = conn.execute("SELECT MAX(cal_date) FROM trade_calendar").fetchone()
        conn.close()
        return row[0] if row and row[0] else None
    except Exception:
        return None


def ensure_calendar_loaded(token: str = "") -> None:
    """
    懒加载 + 自动延期：
      1. 表为空时拉一次（2020 ~ 2030）
      2. 表里最大日期 - 今天 < 6 个月时，自动拉到 +5 年（避免跨年缺数据）
    每次脚本启动时调用一次开销可忽略。
    """
    from datetime import date, timedelta

    n = calendar_count()
    if n == 0:
        logger.info("[trade_calendar] 日历为空，首次拉取 ...")
        update_trade_calendar(token)
        return

    # 检查是否需要延期
    max_date = _calendar_max_date()
    if not max_date:
        return
    try:
        max_d = date.fromisoformat(max_date)
        days_left = (max_d - date.today()).days
        if days_left < 180:  # 6 个月内会用完
            new_end = (date.today() + timedelta(days=365 * 5)).strftime("%Y-%m-%d")
            logger.info("[trade_calendar] 距离日历到期仅 %d 天，自动延期到 %s",
                        days_left, new_end)
            update_trade_calendar(token, start="2020-01-01", end=new_end)
    except Exception as e:
        logger.warning("[trade_calendar] 延期检查异常：%s", e)
