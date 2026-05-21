"""
signal/ranker.py — 信号多维打分

把"5 只票挑哪只"从凭感觉变成有依据。
打分范围 0.0 ~ 1.0，越高越值得入场。

五维：
  25%  信号新鲜度（今日命中 > 昨日 > 更早）
  20%  共振强度（命中策略数）
  20%  趋势对齐（MA20>MA60 + 收盘>MA20 + 量比放大）
  15%  流动性（成交额 / 换手率落在合理区间）
  20%  风险扣分（ST、近 60 日涨幅过大、接近近期高点）
"""

from __future__ import annotations
import pandas as pd
from datetime import date, timedelta


def _prev_workday(latest_date: str) -> str:
    """latest_date 之前的最近一个工作日（不识别节假日）"""
    d = date.fromisoformat(latest_date) - timedelta(days=1)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d.strftime("%Y-%m-%d")


def calc_signal_score(
    df: pd.DataFrame,
    hit_strategies: set | list,
    hit_dates: set | list,
    name: str,
    latest_date: str,
) -> tuple[float, dict]:
    """
    返回 (score, signals_dict)

    df: 单股最近 N 天日线，至少 60 行，列含 close/volume/amount[/turnover]
        要求按 trade_date 升序排列
    hit_strategies: 命中过的策略 id 集合
    hit_dates: 命中过的日期集合
    name: 股票名称（识别 ST）
    latest_date: 'YYYY-MM-DD' 扫描基准日

    signals_dict 包含给上层判断用的辅助字段：
        today_hit (bool)
        high_position (bool, 近期涨幅过大或接近高点)
        gain_60d (float, 近 60 日涨幅 %)
    """
    score = 0.0
    info = {"today_hit": False, "high_position": False, "gain_60d": 0.0}

    if df is None or df.empty or len(df) < 20:
        return 0.0, info

    hit_dates_set = set(hit_dates) if not isinstance(hit_dates, set) else hit_dates

    # ===== 25% 信号新鲜度 =====
    if latest_date in hit_dates_set:
        score += 0.25
        info["today_hit"] = True
    elif _prev_workday(latest_date) in hit_dates_set:
        score += 0.15
    elif hit_dates_set:
        score += 0.05  # 更老的命中

    # ===== 20% 共振强度 =====
    n_strats = len(hit_strategies) if hit_strategies else 0
    score += min(n_strats / 3.0, 1.0) * 0.20

    # ===== 20% 趋势对齐 =====
    close = df["close"].astype(float)

    if len(close) >= 60:
        ma20 = close.rolling(20).mean().iloc[-1]
        ma60 = close.rolling(60).mean().iloc[-1]
        if pd.notna(ma20) and pd.notna(ma60):
            if ma20 > ma60:
                score += 0.10
            if close.iloc[-1] > ma20:
                score += 0.05

    # 量能放大（连续值：今日量 / 20 日均量）
    if "volume" in df.columns and len(df) >= 20:
        vol = df["volume"].astype(float)
        vol_ma20 = vol.rolling(20).mean().iloc[-1]
        if pd.notna(vol_ma20) and vol_ma20 > 0:
            ratio = vol.iloc[-1] / vol_ma20
            # 1.0x → 0  /  1.5x → 0.025  /  2.0x → 0.05  /  >=3.0x → 0.05
            score += max(0.0, min((ratio - 1.0) * 0.05, 0.05))

    # ===== 15% 流动性 =====
    if "amount" in df.columns and len(df) >= 20:
        amt_avg20 = df["amount"].astype(float).tail(20).mean()
        if pd.notna(amt_avg20) and amt_avg20 > 0:
            # 用对数距离衡量"离 1.5 亿成交额的偏离度"，钟形评分
            import math
            log_amt = math.log10(amt_avg20)  # 5e7→7.7, 1.5e8→8.18, 5e8→8.7
            log_target = 8.18  # 1.5 亿元，A 股活跃中盘的甜区
            dist = abs(log_amt - log_target)
            # dist=0 → 0.10, dist=0.5 → 0.05, dist=1.0 → 0
            score += max(0.0, 0.10 - dist * 0.10)

    if "turnover" in df.columns:
        try:
            t = float(df["turnover"].iloc[-1])
            # 钟形：换手率 2% 最佳，0.5%~5% 都给分
            if 0.3 <= t <= 8.0:
                # 离 2 越近分越高
                dist = abs(t - 2.0) / 3.0  # 归一化
                score += max(0.0, 0.05 - dist * 0.025)
        except (ValueError, TypeError):
            pass

    # ===== 20% 风险扣分 =====
    # ST（一票否决）
    if name and "ST" in str(name).upper().replace(" ", ""):
        score -= 0.50

    # 近 60 日涨幅：双向评分
    #   高位扣分：涨多了，追高风险大
    #   低位加分：跌多了，反转空间大（贴合 boll_rv 策略）
    if len(close) >= 60:
        try:
            base = float(close.iloc[-60])
            if base > 0:
                gain_60d = (float(close.iloc[-1]) / base - 1) * 100
                info["gain_60d"] = round(gain_60d, 1)
                if gain_60d > 50:
                    score -= 0.15
                    info["high_position"] = True
                elif gain_60d > 30:
                    score -= 0.05
                    info["high_position"] = True
                elif gain_60d < -25:
                    score += 0.06  # 深跌反弹空间大
                elif gain_60d < -10:
                    score += 0.03  # 浅跌
        except Exception:
            pass

    # 接近窗口最高点（近 120 日内）
    if len(close) >= 30:
        try:
            window_high = float(close.tail(120).max())
            if window_high > 0 and float(close.iloc[-1]) / window_high >= 0.97:
                score -= 0.05
                info["high_position"] = True
        except Exception:
            pass

    # 限制到 [0, 1]
    score = max(0.0, min(score, 1.0))
    return round(score, 3), info


def score_tag(score: float, info: dict) -> str:
    """
    根据 score 和辅助信号，给出一个简短标签（可能为空）。
    优先级：强推 > 不建议 > 信号已老 > 注意高位
    """
    if score >= 0.70:
        return "🔥 强推"
    if score < 0.45:
        return "❌ 不建议"
    if not info.get("today_hit", True):
        return "⏰ 信号已老"
    if info.get("high_position", False):
        return "⚠️ 注意高位"
    if score >= 0.60:
        return "👌 可入"
    return ""
