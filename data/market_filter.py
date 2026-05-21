"""
data/market_filter.py — 市场环境过滤器

两层过滤：
  第一层（当日广度）：上涨占比、大跌股占比、中位涨跌幅
  第二层（趋势判断）：过去N日平均上涨占比，避免在下跌趋势中开仓

没有指数数据，全部基于全市场个股广度计算。
"""

import sqlite3
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DB_PATH

# ── 第一层：当日广度参数 ──
UP_RATIO_MIN        = 0.45    # 上涨家数占比最低门槛
UP_RATIO_WEAK       = 0.40    # 宽松门槛（配合中位数使用）
BIG_DROP_MAX        = 0.20    # 大跌股（跌幅>4%）占比上限
MEDIAN_PCT_MIN      = -0.5    # 中位涨跌幅下限（%）

# ── 第二层：趋势参数 ──
TREND_WINDOW        = 10      # 趋势判断窗口（交易日）
TREND_UP_RATIO_MIN  = 0.42    # 过去N日平均上涨占比门槛
TREND_MEDIAN_MIN    = -0.3    # 过去N日平均中位涨跌幅门槛（%）


class MarketFilter:

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path

    def _compute_breadth_from_pct(self, pct_series: pd.Series, trade_date: str) -> dict | None:
        """从 pct_change Series 计算市场广度指标（不查 DB，纯计算）"""
        pct = pct_series.dropna()
        total = len(pct)
        if total < 100:
            return None

        up        = (pct > 0).sum()
        down      = (pct < 0).sum()
        big_drop  = (pct < -4).sum()

        return {
            "trade_date":     trade_date,
            "total":          total,
            "up_ratio":       round(up / total * 100, 2),
            "down_ratio":     round(down / total * 100, 2),
            "big_drop_ratio": round(big_drop / total * 100, 2),
            "median_pct":     round(pct.median(), 4),
            "mean_pct":       round(pct.mean(), 4),
        }

    def _get_breadth(self, trade_date: str) -> dict | None:
        """计算某交易日的市场广度指标"""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql(
            "SELECT pct_change FROM daily_bars WHERE trade_date = ? AND pct_change IS NOT NULL",
            conn, params=(trade_date,),
        )
        conn.close()
        return self._compute_breadth_from_pct(df["pct_change"], trade_date)

    def _get_trend(self, trade_date: str, window: int = TREND_WINDOW) -> dict:
        """
        计算过去 window 个交易日的市场趋势。
        返回平均上涨占比、平均中位涨跌幅，以及趋势是否健康。

        实现：一次查 window 天的全量 pct_change，groupby 内存里算各日广度。
        旧实现是逐日 N 次查询 + N 次开关连接，慢一个数量级。
        """
        conn = sqlite3.connect(self.db_path)

        # 先取最近 window 个不重复交易日
        recent_dates = pd.read_sql(
            """
            SELECT DISTINCT trade_date FROM daily_bars
            WHERE trade_date <= ?
            ORDER BY trade_date DESC LIMIT ?
            """,
            conn, params=(trade_date, window),
        )["trade_date"].tolist()

        if len(recent_dates) < window // 2:
            conn.close()
            return {"trend_ok": True, "reason": "趋势数据不足，跳过趋势过滤"}

        # 一次性拿这 N 天的全部 pct_change
        placeholders = ",".join(["?"] * len(recent_dates))
        all_df = pd.read_sql(
            f"""
            SELECT trade_date, pct_change FROM daily_bars
            WHERE trade_date IN ({placeholders}) AND pct_change IS NOT NULL
            """,
            conn, params=recent_dates,
        )
        conn.close()

        # 按日分组算广度
        up_ratios   = []
        median_pcts = []
        for td, group in all_df.groupby("trade_date"):
            b = self._compute_breadth_from_pct(group["pct_change"], td)
            if b:
                up_ratios.append(b["up_ratio"])
                median_pcts.append(b["median_pct"])

        if not up_ratios:
            return {"trend_ok": True, "reason": "趋势数据不足，跳过趋势过滤"}

        avg_up_ratio  = sum(up_ratios) / len(up_ratios)
        avg_median    = sum(median_pcts) / len(median_pcts)

        trend_ok = (
            avg_up_ratio >= TREND_UP_RATIO_MIN * 100
            and avg_median >= TREND_MEDIAN_MIN
        )

        return {
            "trend_ok":      trend_ok,
            "avg_up_ratio":  round(avg_up_ratio, 2),
            "avg_median_pct":round(avg_median, 4),
            "window_days":   len(up_ratios),
            "reason": (
                f"近{len(up_ratios)}日均上涨占比{avg_up_ratio:.1f}%，"
                f"均中位涨跌{avg_median:+.2f}%"
            ),
        }

    def get_market_status(self, trade_date: str) -> dict:
        """
        完整市场状态判断（两层过滤）。
        返回字典包含 is_bullish（bool）和详细指标。
        """
        # ── 第一层：当日广度 ──
        breadth = self._get_breadth(trade_date)
        if breadth is None:
            return {
                "trade_date": trade_date,
                "is_bullish": False,
                "reason":     "当日数据不足，无法判断",
            }

        up_ratio    = breadth["up_ratio"] / 100
        big_drop    = breadth["big_drop_ratio"] / 100
        median_pct  = breadth["median_pct"]

        day_ok = (
            (up_ratio >= UP_RATIO_MIN and big_drop <= BIG_DROP_MAX)
            or
            (up_ratio >= UP_RATIO_WEAK and median_pct >= MEDIAN_PCT_MIN)
        )

        if not day_ok:
            return {
                **breadth,
                "is_bullish": False,
                "reason":     "市场偏弱，下跌扩散",
            }

        # ── 第二层：趋势判断 ──
        trend = self._get_trend(trade_date)
        if not trend["trend_ok"]:
            return {
                **breadth,
                "is_bullish":    False,
                "trend_avg_up":  trend.get("avg_up_ratio"),
                "trend_avg_med": trend.get("avg_median_pct"),
                "reason":        f"趋势偏弱 — {trend['reason']}",
            }

        return {
            **breadth,
            "is_bullish":    True,
            "trend_avg_up":  trend.get("avg_up_ratio"),
            "trend_avg_med": trend.get("avg_median_pct"),
            "reason":        f"上涨家数占比高且趋势健康 — {trend['reason']}",
        }


# ── 单例工厂 ──
_instance = None

def get_market_filter() -> MarketFilter:
    global _instance
    if _instance is None:
        _instance = MarketFilter()
    return _instance
