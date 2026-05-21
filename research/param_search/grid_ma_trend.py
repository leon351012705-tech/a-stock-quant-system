"""
research/param_search/grid_ma_trend.py — 任务1 / 均线多头排列 ma_trend 参数网格（补完决策1）

补完之前略过的策略 grid。ma_trend 是动量类（MA 多头排列 + MA5 上穿 MA20 入场），
跟 MACD 同一性质（滞后确认指标），预期：参数无杠杆、单独当选股策略不行。

网格：fast {3, 5, 8} × mid {15, 20, 25} × slow {45, 60, 80} × use_entry_filter {True, False}
     = 3 × 3 × 3 × 2 = 54 组
baseline: fast=5, mid=20, slow=60, use_entry_filter=True（共振配置默认）

出场用 _common 里的 sys（-5%硬止损 + -5%移动止盈 + 20日）——跟 MACD 那轮同口径，方便对比。
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from research.param_search._runner import run_grid_search   # noqa: E402


def ma_trend_buy_signal(p: dict, fast: int, mid: int, slow: int, use_entry_filter: bool) -> np.ndarray:
    """向量化等价 research/strategies/ma_trend.py 的 buy 分支。"""
    close = pd.Series(p["close"])
    ma_f  = close.rolling(fast).mean()
    ma10  = close.rolling(10).mean()
    ma_m  = close.rolling(mid).mean()
    ma_s  = close.rolling(slow).mean()
    bullish     = (ma_f > ma10) & (ma10 > ma_m) & (ma_m > ma_s)
    price_above = close > ma_f
    cross_up    = (ma_f > ma_m) & (ma_f.shift(1) <= ma_m.shift(1))
    buy = bullish & price_above
    if use_entry_filter:
        buy = buy & cross_up
    return buy.fillna(False).to_numpy()


GRID = {
    "fast": [3, 5, 8],
    "mid":  [15, 20, 25],
    "slow": [45, 60, 80],
    "use_entry_filter": [True, False],
}   # 3 × 3 × 3 × 2 = 54

BASELINE = {"fast": 5, "mid": 20, "slow": 60, "use_entry_filter": True}


if __name__ == "__main__":
    run_grid_search("ma_trend", ma_trend_buy_signal, GRID, BASELINE)
