"""
data/market_filter_v2.py — 数据校准后的 market_filter（任务3 实验记录）

⚠️ 测过后不推荐落地：在 OOS 上跟 v1 决策完全一致（326/660 通过，Δret +1.58pp 相同），
   在 IS 上反而比 v1 少通过 15 条 +3.18% 的好信号。
   保留此文件作为"尝试过、与 v1 等价"的实验文档。
   研究脚本：research/param_search/market_filter_calibration.py、eval_mfilter_v2.py。

跟 data/market_filter.py 并存，**不动原版**。要切：把 signals/scanner.py 里那行
  from data.market_filter import get_market_filter
改成
  from data.market_filter_v2 import get_market_filter
即可。get_market_status 接口完全一致。

阈值变动（基于 research/param_search/market_filter_calibration.py 的 1559+660 条
共振信号实证回归 + 单维敏感性扫描）：

  原阈值                            新阈值              理由
  UP_RATIO_MIN        0.45     →    0.40              OOS 0.40 比 0.45 Δret +1.21 vs +0.39
  UP_RATIO_WEAK       0.40     →    （废除）          原 OR 分支冗余；新规则用单一 AND
  BIG_DROP_MAX        0.20     →    （删除！）        IS/OOS 都是负贡献——大跌股多的日子
                                                     反而是均值回归的好机会（IS Δ -3.16,
                                                     OOS Δ -2.96）
  MEDIAN_PCT_MIN      -0.5     →    -0.5              IS 想紧/OOS 想松，保守保留
  TREND_UP_RATIO_MIN  0.42     →    0.45              IS Δ +0.91 vs 0.42 的 +0.48；OOS 同
  TREND_MEDIAN_MIN    -0.3     →    -0.3              IS/OOS 方向反，保守保留

新规则（比原版简洁很多）：
  day_ok   = up_ratio >= 0.40 AND median_pct >= -0.5
  trend_ok = avg_up_10d >= 0.45 AND avg_median_10d >= -0.3
  is_bullish = day_ok AND trend_ok
"""

import sqlite3
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DB_PATH

# ── 第一层：当日广度 ──
UP_RATIO_MIN        = 0.40    # ↓ 从 0.45 降到 0.40
MEDIAN_PCT_MIN      = -0.5    # 保留
# UP_RATIO_WEAK 已废除（合并到单一规则里）
# BIG_DROP_MAX 已删除（反向作用，详见上方说明）

# ── 第二层：10 日趋势 ──
TREND_WINDOW        = 10
TREND_UP_RATIO_MIN  = 0.45    # ↑ 从 0.42 升到 0.45
TREND_MEDIAN_MIN    = -0.3    # 保留


class MarketFilter:

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path

    def _compute_breadth_from_pct(self, pct_series: pd.Series, trade_date: str) -> dict | None:
        """从 pct_change Series 计算市场广度指标（纯计算，不查 DB）"""
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
            "big_drop_ratio": round(big_drop / total * 100, 2),     # 仍记录，但不再当过滤条件
            "median_pct":     round(pct.median(), 4),
            "mean_pct":       round(pct.mean(), 4),
        }

    def _get_breadth(self, trade_date: str) -> dict | None:
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql(
            "SELECT pct_change FROM daily_bars WHERE trade_date = ? AND pct_change IS NOT NULL",
            conn, params=(trade_date,),
        )
        conn.close()
        return self._compute_breadth_from_pct(df["pct_change"], trade_date)

    def _get_trend(self, trade_date: str, window: int = TREND_WINDOW) -> dict:
        conn = sqlite3.connect(self.db_path)
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

        placeholders = ",".join(["?"] * len(recent_dates))
        all_df = pd.read_sql(
            f"""
            SELECT trade_date, pct_change FROM daily_bars
            WHERE trade_date IN ({placeholders}) AND pct_change IS NOT NULL
            """,
            conn, params=recent_dates,
        )
        conn.close()

        up_ratios, median_pcts = [], []
        for td, group in all_df.groupby("trade_date"):
            b = self._compute_breadth_from_pct(group["pct_change"], td)
            if b:
                up_ratios.append(b["up_ratio"])
                median_pcts.append(b["median_pct"])
        if not up_ratios:
            return {"trend_ok": True, "reason": "趋势数据不足，跳过趋势过滤"}

        avg_up_ratio = sum(up_ratios) / len(up_ratios)
        avg_median   = sum(median_pcts) / len(median_pcts)
        trend_ok = (avg_up_ratio >= TREND_UP_RATIO_MIN * 100
                    and avg_median >= TREND_MEDIAN_MIN)
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
        """完整市场状态判断（两层过滤）。接口与 v1 一致。"""
        breadth = self._get_breadth(trade_date)
        if breadth is None:
            return {
                "trade_date": trade_date, "is_bullish": False,
                "reason":     "当日数据不足，无法判断",
            }

        up_ratio   = breadth["up_ratio"] / 100
        median_pct = breadth["median_pct"]

        # 新规则：单一 AND 条件，去掉 OR 分支和 big_drop
        day_ok = (up_ratio >= UP_RATIO_MIN and median_pct >= MEDIAN_PCT_MIN)

        if not day_ok:
            return {
                **breadth, "is_bullish": False,
                "reason": (f"当日广度偏弱（up={up_ratio*100:.0f}%, "
                           f"median={median_pct:+.2f}%）"),
            }

        trend = self._get_trend(trade_date)
        if not trend["trend_ok"]:
            return {
                **breadth, "is_bullish": False,
                "trend_avg_up":  trend.get("avg_up_ratio"),
                "trend_avg_med": trend.get("avg_median_pct"),
                "reason":        f"趋势偏弱 — {trend['reason']}",
            }

        return {
            **breadth, "is_bullish": True,
            "trend_avg_up":  trend.get("avg_up_ratio"),
            "trend_avg_med": trend.get("avg_median_pct"),
            "reason":        f"广度+趋势均健康 — {trend['reason']}",
        }


# ── 单例工厂（与 v1 接口一致） ──
_instance = None

def get_market_filter() -> MarketFilter:
    global _instance
    if _instance is None:
        _instance = MarketFilter()
    return _instance
