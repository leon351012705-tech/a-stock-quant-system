"""
research/strategies/ssb_bounce.py
策略：趋势回踩确认 V5（Trend-Pullback-Bounce）

核心逻辑转变（与V1-V4完全不同）：
  V1-V4: 在下跌趋势中抄底 → 胜率 ~41%
  V5:    在上升趋势中买回调 → 目标胜率 ≥55%

入场逻辑：
  ① 趋势向上：MA20 > MA60，且 close > MA60（牛股特征）
  ② 短期回调：近3天内至少2天收阴，回调到 MA20 附近
  ③ 今日企稳：收阳线，收盘在日内上半部
  ④ 缩量回调：回调期间量能萎缩（健康回调而非放量出逃）
  ⑤ 今日放量：今日成交量 > 昨日（资金回流）

核心原理：
  上升趋势中 MA20 是动态支撑位
  回调到 MA20 附近 + 缩量 = 正常获利回吐（不是趋势反转）
  当天放量收阳 = 主力重新介入
  次日买入骑趋势延续
"""

import pandas as pd
import numpy as np


DEFAULT_PARAMS = {
    # ── 选股过滤 ──
    "min_price": 3.0,
    "min_amount_20d": 2000e4,
    "min_turnover": 0.5,
    "min_history_days": 60,

    # ── 条件A：趋势向上 ──
    "ma_fast": 20,
    "ma_slow": 60,
    "require_ma_up": True,      # MA20 > MA60

    # ── 条件B：短期回调到 MA20 附近 ──
    "pullback_days": 3,         # 回看3天
    "min_red_days": 1,          # 至少1天收阴（有回调过程）
    "ma_touch_upper": 1.03,     # 价格 ≤ MA20 * 1.03（贴近MA20）
    "ma_touch_lower": 0.97,     # 价格 ≥ MA20 * 0.97（不能跌穿太多）

    # ── 条件C：今日企稳收阳 ──
    "min_close_position": 0.5,  # 收盘在日内上半部
    "min_today_gain": 0.1,      # 今日至少微涨
    "max_today_gain": 5.0,      # 不能涨太多

    # ── 条件D：回调期间缩量 ──
    "pullback_vol_ratio": 0.85, # 近3日均量 < 20日均量 * 0.85

    # ── 条件E：今日放量（资金回流）──
    "today_vol_increase": 1.05, # 今日量 > 昨日量

    # ── 条件F：不在高位（防追高）──
    "max_above_ma60_pct": 0.30, # 价格不超过 MA60 的130%
}


def generate_signals(df: pd.DataFrame, params: dict = None) -> pd.Series:
    """
    生成趋势回踩确认信号 V5。
    """
    if params is None:
        params = DEFAULT_PARAMS.copy()

    signals = pd.Series(0, index=df.index)

    if len(df) < params["min_history_days"]:
        return signals

    close  = df["close"].values
    open_  = df["open"].values
    high   = df["high"].values
    low    = df["low"].values
    volume = df["volume"].values

    # pct_change
    if "pct_change" in df.columns:
        pct = df["pct_change"].values
    else:
        pct = np.zeros(len(df))
        pct[1:] = (close[1:] / close[:-1] - 1) * 100

    # turnover
    has_turnover = "turnover" in df.columns and df["turnover"].notna().any()
    turnover = df["turnover"].values if has_turnover else np.full(len(df), 999.0)

    # amount
    has_amount = "amount" in df.columns and df["amount"].notna().any()
    amount = df["amount"].values if has_amount else np.full(len(df), 1e9)

    ma_fast = params["ma_fast"]
    ma_slow = params["ma_slow"]

    # 均线
    s_close = df["close"]
    ma_fast_arr = s_close.rolling(ma_fast, min_periods=ma_fast).mean().values
    ma_slow_arr = s_close.rolling(ma_slow, min_periods=ma_slow).mean().values

    # 成交量均线
    s_vol = df["volume"]
    vol_ma20 = s_vol.rolling(20, min_periods=20).mean().values

    # 20日均成交额
    if has_amount:
        amount_ma20 = df["amount"].rolling(20, min_periods=20).mean().values
    else:
        amount_ma20 = np.full(len(df), 1e9)

    pullback_days = params["pullback_days"]
    min_start = max(ma_slow, 20, pullback_days + 1)

    # 涨跌停阈值：优先 df.attrs（scanner 透传的板块/ST 感知值），
    # 其次 params['limit_pct']，最后兜底 9.5（旧行为）
    limit_pct = df.attrs.get("limit_pct", params.get("limit_pct", 9.5))
    # 0.1 缓冲（跟 limit_rules.is_limit_move 保持一致），不再用固定 0.5
    # 旧的 0.5 对 ST（5% 涨停）等于 10% 缓冲，会把 +4.5%~+4.99% 范围误判为涨停
    limit_threshold = limit_pct - 0.1

    for i in range(min_start, len(df)):

        # ===== 选股过滤 =====
        if close[i] < params["min_price"]:
            continue
        if amount_ma20[i] < params["min_amount_20d"]:
            continue
        if turnover[i] < params["min_turnover"]:
            continue
        if pct[i] > limit_threshold or pct[i] < -limit_threshold:
            continue
        if volume[i] <= 0:
            continue

        # ===== 条件A：趋势向上 =====
        if np.isnan(ma_fast_arr[i]) or np.isnan(ma_slow_arr[i]):
            continue

        # MA20 > MA60（趋势确认）
        if params.get("require_ma_up", True):
            if ma_fast_arr[i] <= ma_slow_arr[i]:
                continue

        # 收盘价 > MA60（在趋势之上）
        if close[i] < ma_slow_arr[i]:
            continue

        # ===== 条件B：短期回调到 MA20 附近 =====

        # 近N天至少有min_red_days天收阴（确认有过回调）
        red_days = 0
        for j in range(i - pullback_days, i):
            if j >= 0 and close[j] < open_[j]:
                red_days += 1
        if red_days < params["min_red_days"]:
            continue

        # 价格在 MA20 附近（贴近支撑位）
        if ma_fast_arr[i] <= 0:
            continue
        price_vs_ma = close[i] / ma_fast_arr[i]
        if price_vs_ma > params["ma_touch_upper"]:
            continue
        if price_vs_ma < params["ma_touch_lower"]:
            continue

        # ===== 条件C：今日企稳收阳 =====
        # 今日收阳
        if close[i] <= open_[i]:
            continue

        # 涨幅合理
        if pct[i] < params["min_today_gain"]:
            continue
        if pct[i] > params["max_today_gain"]:
            continue

        # 收盘在日内上半部
        day_range = high[i] - low[i]
        if day_range <= 0:
            continue
        close_position = (close[i] - low[i]) / day_range
        if close_position < params["min_close_position"]:
            continue

        # ===== 条件D：回调期间缩量 =====
        if not np.isnan(vol_ma20[i]) and vol_ma20[i] > 0:
            # 近pullback_days日均量 < 20日均量 * 阈值
            recent_vol = 0
            count = 0
            for j in range(i - pullback_days, i):
                if j >= 0:
                    recent_vol += volume[j]
                    count += 1
            if count > 0:
                avg_recent = recent_vol / count
                if avg_recent >= vol_ma20[i] * params["pullback_vol_ratio"]:
                    continue

        # ===== 条件E：今日放量 =====
        if i < 1 or volume[i - 1] <= 0:
            continue
        if volume[i] / volume[i - 1] < params["today_vol_increase"]:
            continue

        # ===== 条件F：不在高位 =====
        if ma_slow_arr[i] > 0:
            above_pct = close[i] / ma_slow_arr[i] - 1
            if above_pct > params["max_above_ma60_pct"]:
                continue

        # ===== 全部通过 =====
        signals.iloc[i] = 1

    return signals


def get_signal_score(df: pd.DataFrame, signal_idx: int,
                     params: dict = None) -> float:
    """信号强度评分"""
    if params is None:
        params = DEFAULT_PARAMS.copy()

    i = signal_idx
    close  = df["close"].values
    open_  = df["open"].values
    high   = df["high"].values
    low    = df["low"].values
    volume = df["volume"].values

    ma_fast = params.get("ma_fast", 20)
    ma_fast_arr = df["close"].rolling(ma_fast, min_periods=ma_fast).mean().values
    vol_ma20 = df["volume"].rolling(20, min_periods=20).mean().values

    # 维度1：MA20附近精度（越贴近MA20越好，权重0.3）
    if not np.isnan(ma_fast_arr[i]) and ma_fast_arr[i] > 0:
        deviation = abs(close[i] / ma_fast_arr[i] - 1)
        ma_score = max(0, 1 - deviation / 0.03)  # 偏离0%得满分，偏离3%得0分
    else:
        ma_score = 0

    # 维度2：放量强度（权重0.25）
    if i >= 1 and volume[i - 1] > 0:
        vol_surge = volume[i] / volume[i - 1]
        vol_score = min((vol_surge - 1.0) / 0.5, 1.0)
        vol_score = max(vol_score, 0)
    else:
        vol_score = 0

    # 维度3：阳线质量（权重0.25）
    day_range = high[i] - low[i]
    if day_range > 0:
        body = close[i] - open_[i]
        body_score = min(max(body / day_range, 0) / 0.6, 1.0)
    else:
        body_score = 0

    # 维度4：趋势强度 MA20与MA60的差距（权重0.2）
    ma_slow_arr = df["close"].rolling(60, min_periods=60).mean().values
    if (not np.isnan(ma_fast_arr[i]) and not np.isnan(ma_slow_arr[i])
            and ma_slow_arr[i] > 0):
        trend_strength = ma_fast_arr[i] / ma_slow_arr[i] - 1
        trend_score = min(max(trend_strength / 0.1, 0), 1.0)
    else:
        trend_score = 0

    score = ma_score * 0.3 + vol_score * 0.25 + body_score * 0.25 + trend_score * 0.2
    return round(score, 4)