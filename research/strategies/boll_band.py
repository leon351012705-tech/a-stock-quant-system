"""
【研究层】strategies/boll_band.py — 布林带策略

策略逻辑（双模式，可切换）：

模式A：均值回归（超跌反弹）
  买入：收盘价触碰下轨（跌破下轨后收回）且成交量放大
  卖出：收盘价触碰中轨（MA20）或上轨

模式B：突破趋势（布林带收口后放量突破上轨）
  买入：布林带从收窄状态放量向上突破上轨
  卖出：收盘价跌回中轨（MA20）下方

参数（与同花顺默认一致）：
  - 中轨：20日移动平均线
  - 上轨：中轨 + 2倍标准差
  - 下轨：中轨 - 2倍标准差

使用方式：
  from research.strategies.boll_band import generate_signals
  signals = generate_signals(df, mode='reversion')   # 均值回归
  signals = generate_signals(df, mode='breakout')    # 趋势突破
"""

import pandas as pd
import numpy as np


def calc_boll(
    close: pd.Series,
    period: int = 20,
    std_mult: float = 2.0,
) -> pd.DataFrame:
    """
    计算布林带指标。

    参数：
        close    : 收盘价 Series
        period   : 中轨均线周期，默认20
        std_mult : 标准差倍数，默认2

    返回：
        DataFrame，包含 mid / upper / lower / bandwidth / pct_b 列
    """
    mid   = close.rolling(window=period).mean()
    std   = close.rolling(window=period).std()
    upper = mid + std_mult * std
    lower = mid - std_mult * std

    # 带宽：衡量布林带开口程度，值越小代表越收窄（蓄势待发）
    bandwidth = (upper - lower) / mid * 100

    # %B：价格在布林带内的相对位置，0=下轨，1=上轨，0.5=中轨
    pct_b = (close - lower) / (upper - lower)

    return pd.DataFrame({
        "mid"      : mid,
        "upper"    : upper,
        "lower"    : lower,
        "bandwidth": bandwidth,
        "pct_b"    : pct_b,
    })


def generate_signals(
    df: pd.DataFrame,
    period: int = 20,
    std_mult: float = 2.0,
    mode: str = "reversion",
) -> pd.Series:
    """
    生成布林带交易信号。

    参数：
        df       : 日线数据，需包含 close / volume 列
        period   : 布林带周期，默认20
        std_mult : 标准差倍数，默认2
        mode     : 'reversion'=均值回归 / 'breakout'=趋势突破

    返回：
        signals Series（1=买入，-1=卖出，0=无操作），与 df 等长
    """
    boll   = calc_boll(df["close"], period, std_mult)
    close  = df["close"]
    volume = df["volume"] if "volume" in df.columns else pd.Series(1, index=df.index)
    vol_ma = volume.rolling(window=20).mean()

    signals = pd.Series(0, index=df.index)

    for i in range(period + 1, len(df)):
        c      = close.iloc[i]
        c_pre  = close.iloc[i - 1]
        mid    = boll["mid"].iloc[i]
        upper  = boll["upper"].iloc[i]
        lower  = boll["lower"].iloc[i]
        bw     = boll["bandwidth"].iloc[i]
        bw_pre = boll["bandwidth"].iloc[i - 5] if i >= period + 5 else bw
        v      = volume.iloc[i]
        v_avg  = vol_ma.iloc[i]

        if mode == "reversion":
            # ── 均值回归模式 ──
            # 买入：昨日跌破下轨，今日收回下轨上方（反弹确认）
            touched_lower = (c_pre < boll["lower"].iloc[i - 1])
            bounced_back  = (c > lower)

            # 卖出：收盘价超过中轨（止盈）或跌破下轨更深（止损）
            reach_mid  = (c >= mid)
            deep_break = (c < lower * 0.97)

            # 纯信号输出，不维护 position
            if touched_lower and bounced_back:
                signals.iloc[i] = 1
            elif reach_mid or deep_break:
                signals.iloc[i] = -1

        elif mode == "breakout":
            # ── 趋势突破模式 ──
            # 买入：布林带从收窄转为放宽 + 收盘突破上轨 + 成交量放大
            bw_expanding = (bw > bw_pre * 1.1)   # 带宽比5日前扩大10%
            break_upper  = (c > upper and c_pre <= boll["upper"].iloc[i - 1])
            vol_surge    = (v > v_avg * 1.2) if v_avg > 0 else True

            # 卖出：收盘跌回中轨下方
            below_mid = (c < mid)

            buy_cond  = break_upper and (bw_expanding or vol_surge)
            sell_cond = below_mid

            # 纯信号输出，不维护 position
            if sell_cond:
                signals.iloc[i] = -1
            elif buy_cond:
                signals.iloc[i] = 1

        else:
            raise ValueError(f"Unsupported mode: {mode}")

    return signals


def get_strategy_info() -> dict:
    """返回策略基本信息。"""
    return {
        "name"       : "布林带策略",
        "version"    : "1.0",
        "description": "均值回归（触下轨反弹）或趋势突破（放量突破上轨）双模式",
        "params"     : {"period": 20, "std_mult": 2.0},
        "author"     : "quant_system",
    }