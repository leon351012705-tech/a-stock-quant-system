"""
【研究层】strategies/macd_cross.py — MACD 金叉/死叉策略

策略逻辑：
  - 买入：DIFF 从下方上穿 DEA（金叉），且 DIFF < 0（零轴下方金叉更可靠）
  - 卖出：DIFF 从上方下穿 DEA（死叉）

参数（与同花顺默认一致，方便对比）：
  - 短期EMA：12日
  - 长期EMA：26日
  - 信号线DEA：9日

输出：标准化信号 Series（1=买入，-1=卖出，0=持有）
      供回测层直接调用
"""

import pandas as pd
import numpy as np


def calc_macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """
    计算 MACD 指标。

    参数：
        close  : 收盘价 Series
        fast   : 短期EMA周期，默认12
        slow   : 长期EMA周期，默认26
        signal : 信号线周期，默认9

    返回：
        DataFrame，包含 ema_fast / ema_slow / diff / dea / macd 列
    """
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    diff     = ema_fast - ema_slow
    dea      = diff.ewm(span=signal, adjust=False).mean()
    macd_bar = (diff - dea) * 2

    return pd.DataFrame({
        "ema_fast": ema_fast,
        "ema_slow": ema_slow,
        "diff"    : diff,
        "dea"     : dea,
        "macd"    : macd_bar,
    })


def generate_signals(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    zero_axis_filter: bool = True,
) -> pd.Series:
    """
    生成 MACD 金叉/死叉交易信号。

    参数：
        df               : 日线数据，需包含 close 列
        fast             : 短期EMA周期
        slow             : 长期EMA周期
        signal           : 信号线周期
        zero_axis_filter : True=只在零轴下方金叉买入（过滤高位金叉）

    返回：
        signals Series（1=买入，-1=卖出，0=无操作），与 df 等长
    """
    macd_df = calc_macd(df["close"], fast, slow, signal)
    diff = macd_df["diff"]
    dea  = macd_df["dea"]

    signals = pd.Series(0, index=df.index)

    for i in range(1, len(df)):
        prev_diff = diff.iloc[i - 1]
        curr_diff = diff.iloc[i]
        prev_dea  = dea.iloc[i - 1]
        curr_dea  = dea.iloc[i]

        golden_cross = (prev_diff <= prev_dea) and (curr_diff > curr_dea)
        death_cross  = (prev_diff >= prev_dea) and (curr_diff < curr_dea)

        buy_cond = golden_cross
        if zero_axis_filter:
            buy_cond = buy_cond and (curr_diff < 0)

        sell_cond = death_cross

        if sell_cond:
            signals.iloc[i] = -1
        elif buy_cond:
            signals.iloc[i] = 1

    return signals


def get_strategy_info() -> dict:
    """返回策略基本信息，供信号层调用时展示。"""
    return {
        "name"       : "MACD金叉策略",
        "version"    : "1.0",
        "description": "DIFF上穿DEA金叉买入，下穿死叉卖出",
        "params"     : {"fast": 12, "slow": 26, "signal": 9},
        "author"     : "quant_system",
    }