"""
【研究层】strategies/ma_trend.py — 均线多头排列策略

策略逻辑：
  买入条件（全部满足）：
    1. MA5 > MA10 > MA20 > MA60（多头排列）
    2. 收盘价 > MA5（价格在均线上方，不追高）
    3. MA5 今日上穿 MA20（趋势刚启动，而不是已经走了一大段）

  卖出条件（任一满足）：
    1. 收盘价跌破 MA20（趋势破坏）
    2. MA5 下穿 MA20（死叉，趋势反转确认）

设计理由：
  - 均线多头排列是趋势跟踪的经典信号，胜在简单稳定
  - MA5上穿MA20作为入场点，避免追高买入已走很长的趋势
  - 跌破MA20止损，控制回撤
"""

import pandas as pd


def calc_ma(close: pd.Series, periods: list = [5, 10, 20, 60]) -> pd.DataFrame:
    """
    计算多条均线。

    参数：
        close   : 收盘价 Series
        periods : 均线周期列表

    返回：
        DataFrame，列名为 ma5 / ma10 / ma20 / ma60
    """
    result = {}
    for p in periods:
        result[f"ma{p}"] = close.rolling(window=p).mean()
    return pd.DataFrame(result)


def generate_signals(
    df: pd.DataFrame,
    fast: int = 5,
    mid: int = 20,
    slow: int = 60,
    use_entry_filter: bool = True,
) -> pd.Series:
    """
    生成均线多头排列交易信号。

    参数：
        df               : 日线数据，需包含 close 列
        fast             : 快线周期，默认5
        mid              : 中线周期，默认20（也是止损线）
        slow             : 慢线周期，默认60
        use_entry_filter : True=用快线上穿中线作为入场过滤（避免追高）

    返回：
        signals Series（1=买入，-1=卖出，0=无操作），与 df 等长
    """
    ma_df = calc_ma(df["close"], periods=[fast, 10, mid, slow])
    ma_f  = ma_df[f"ma{fast}"]    # MA5
    ma_m  = ma_df[f"ma{mid}"]     # MA20
    ma_s  = ma_df[f"ma{slow}"]    # MA60
    ma_10 = ma_df["ma10"]
    close = df["close"]

    signals = pd.Series(0, index=df.index)

    for i in range(slow + 1, len(df)):
        mf  = ma_f.iloc[i]
        m10 = ma_10.iloc[i]
        mm  = ma_m.iloc[i]
        ms  = ma_s.iloc[i]
        c   = close.iloc[i]

        # ── 多头排列检测 ──
        is_bullish_arrangement = (mf > m10 > mm > ms)

        # ── 价格在 MA5 上方（不买回调太深的） ──
        price_above_ma5 = (c > mf)

        # ── MA5 上穿 MA20（今日 MA5>MA20，昨日 MA5<=MA20）──
        ma5_cross_ma20 = (
            ma_f.iloc[i] > ma_m.iloc[i] and
            ma_f.iloc[i - 1] <= ma_m.iloc[i - 1]
        )

        # ── 卖出条件 ──
        price_below_ma20 = (c < mm)

        # 严格按你注释里的定义：MA5 下穿 MA20（死叉确认）
        ma5_cross_down_ma20 = (
            ma_f.iloc[i] < ma_m.iloc[i] and
            ma_f.iloc[i - 1] >= ma_m.iloc[i - 1]
        )

        # ── 买入信号 ──
        buy_cond = is_bullish_arrangement and price_above_ma5
        if use_entry_filter:
            buy_cond = buy_cond and ma5_cross_ma20

        # ── 卖出信号 ──
        sell_cond = price_below_ma20 or ma5_cross_down_ma20

        if buy_cond:
            signals.iloc[i] = 1
        elif sell_cond:
            signals.iloc[i] = -1

    return signals


def get_strategy_info() -> dict:
    """返回策略基本信息。"""
    return {
        "name"       : "均线多头排列策略",
        "version"    : "1.0",
        "description": "MA5上穿MA20触发，MA5>MA10>MA20>MA60多头排列，跌破MA20止损",
        "params"     : {"fast": 5, "mid": 20, "slow": 60},
        "author"     : "quant_system",
    }