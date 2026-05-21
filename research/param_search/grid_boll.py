"""
research/param_search/grid_boll.py — 任务1 / 布林带（均值回归 reversion 模式）参数网格

为什么先做布林带：共振规则要求 boll_rv 必须命中（"硬门"），它对最终共振池表现杠杆最大；
且均值回归（触下轨反弹）在 A 股有可能有正期望，是 4 个里最可能有"甜区"的。

评估的是 run_resonance_backtest 里实际用的那支：boll_band.generate_signals(df, mode="reversion")
  买入：昨日收盘 < 昨日下轨 且 今日收盘 > 今日下轨（跌破下轨后收回）
  出场：沿用 _common 的 -5%硬止损 / -5%移动止盈 / 持满20日（与共振回测一致；不用策略自带的"到中轨止盈"）

可调参数：period（中轨均线周期）、std_mult（标准差倍数）。baseline = 20 / 2.0（同花顺默认）。

⚠️ 用固定出场而非策略自带的"到中轨止盈"评估，是为了跟系统实际用法 + MACD 那轮口径一致。
   若结果有意思，可再补一版"策略自带出场"的对比。
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from research.param_search._runner import run_grid_search   # noqa: E402


def boll_rv_buy_signal(p: dict, period: int, std_mult: float) -> np.ndarray:
    """向量化的布林带均值回归买入信号；逻辑等价 boll_band.generate_signals(mode='reversion') 的 buy 分支。"""
    close = pd.Series(p["close"])
    mid = close.rolling(period).mean()
    std = close.rolling(period).std()              # ddof=1，与原策略 .std() 一致
    lower = mid - std_mult * std
    touched = close.shift(1) < lower.shift(1)      # 昨日收盘跌破昨日下轨
    bounced = close > lower                        # 今日收盘收回下轨上方
    buy = touched & bounced
    return buy.fillna(False).to_numpy()


GRID = {
    "period":   [8, 10, 12, 15, 18, 20, 25, 30],
    "std_mult": [1.4, 1.6, 1.8, 2.0, 2.2, 2.5],
}   # 8 × 6 = 48 组

BASELINE = {"period": 20, "std_mult": 2.0}


if __name__ == "__main__":
    run_grid_search("boll_rv", boll_rv_buy_signal, GRID, BASELINE)
