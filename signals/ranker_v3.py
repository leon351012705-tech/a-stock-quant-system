"""
signals/ranker_v3.py — 数据驱动的 ranker v3（任务2 第二次尝试）

跟 ranker_v2 不同：这次只动一个真正有数据支持的维度——**gain_60d**。
保留 v1 的其他维度（不动其他没坏的东西），让 A/B 测试更干净。

数据依据（research/param_search/recent_signals_audit.py，2026-04-01~2026-05-13）：
  gain_60d 分组      n   胜率    均收益
  深跌 <-25%        21   90.5%   +4.40%   ← 实战甜区，必须重权
  浅跌 -25~-10%     46   50.0%   +0.55%
  小跌 -10~0%       18   61.1%   +2.70%
  小涨 0~+10%        4   50.0%   +4.17%
  上涨 >+10%         6   50.0%   +2.50%   ← 追高场景，必须降权

v1 的 gain_60d 加分太弱（深跌只 +0.06），v3 给它 +0.20。
对照 v1，其他 4 个维度（freshness/resonance/trend/liquidity）一字不变。

不动 ranker.py 原版。要切换，把 scanner.py 的 import 改成 ranker_v3。
"""

from __future__ import annotations
import math
import pandas as pd
from datetime import date, timedelta


def _prev_workday(latest_date: str) -> str:
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
    接口与 ranker.py.calc_signal_score 一致。v3 唯一差别：gain_60d 维度重权。
    """
    score = 0.0
    info = {"today_hit": False, "high_position": False, "gain_60d": 0.0, "version": "v3"}

    if df is None or df.empty or len(df) < 20:
        return 0.0, info

    hit_dates_set = set(hit_dates) if not isinstance(hit_dates, set) else hit_dates

    # ===== 25% 信号新鲜度（同 v1） =====
    if latest_date in hit_dates_set:
        score += 0.25
        info["today_hit"] = True
    elif _prev_workday(latest_date) in hit_dates_set:
        score += 0.15
    elif hit_dates_set:
        score += 0.05

    # ===== 20% 共振强度（同 v1） =====
    n_strats = len(hit_strategies) if hit_strategies else 0
    score += min(n_strats / 3.0, 1.0) * 0.20

    # ===== 20% 趋势对齐（同 v1） =====
    close = df["close"].astype(float)
    if len(close) >= 60:
        ma20 = close.rolling(20).mean().iloc[-1]
        ma60 = close.rolling(60).mean().iloc[-1]
        if pd.notna(ma20) and pd.notna(ma60):
            if ma20 > ma60:
                score += 0.10
            if close.iloc[-1] > ma20:
                score += 0.05
    if "volume" in df.columns and len(df) >= 20:
        vol = df["volume"].astype(float)
        vol_ma20 = vol.rolling(20).mean().iloc[-1]
        if pd.notna(vol_ma20) and vol_ma20 > 0:
            ratio = vol.iloc[-1] / vol_ma20
            score += max(0.0, min((ratio - 1.0) * 0.05, 0.05))

    # ===== 15% 流动性（同 v1） =====
    if "amount" in df.columns and len(df) >= 20:
        amt_avg20 = df["amount"].astype(float).tail(20).mean()
        if pd.notna(amt_avg20) and amt_avg20 > 0:
            log_amt = math.log10(amt_avg20)
            dist = abs(log_amt - 8.18)
            score += max(0.0, 0.10 - dist * 0.10)
    if "turnover" in df.columns:
        try:
            t = float(df["turnover"].iloc[-1])
            if 0.3 <= t <= 8.0:
                dist = abs(t - 2.0) / 3.0
                score += max(0.0, 0.05 - dist * 0.025)
        except (ValueError, TypeError):
            pass

    # ===== 风险扣分 / 加分（v3 重写的核心部分） =====
    # ST：硬过滤（同 v1，v1 是 -0.50 软扣；v3 直接 0）
    if name and "ST" in str(name).upper().replace(" ", ""):
        score -= 0.50

    # 60 日涨幅：v3 阶梯式重权
    if len(close) >= 60:
        try:
            base = float(close.iloc[-60])
            if base > 0:
                gain_60d = (float(close.iloc[-1]) / base - 1) * 100
                info["gain_60d"] = round(gain_60d, 1)
                # 深跌甜区：实证 90% 胜率，必须强烈加分
                if gain_60d < -25:
                    score += 0.20                  # v1 是 +0.06
                elif gain_60d < -10:
                    score += 0.05                  # v1 是 +0.03
                elif gain_60d < 0:
                    score += 0.02                  # v1 没区分
                elif gain_60d < 10:
                    score -= 0.05                  # v1 没扣
                    info["high_position"] = True
                elif gain_60d < 30:
                    score -= 0.20                  # v1 是 -0.05
                    info["high_position"] = True
                elif gain_60d < 50:
                    score -= 0.35                  # v1 是 -0.05
                    info["high_position"] = True
                else:
                    score -= 0.50                  # v1 是 -0.15
                    info["high_position"] = True
        except Exception:
            pass

    # 距 120 日最高（同 v1 软扣分）
    if len(close) >= 30:
        try:
            window_high = float(close.tail(120).max())
            if window_high > 0 and float(close.iloc[-1]) / window_high >= 0.97:
                score -= 0.05
                info["high_position"] = True
        except Exception:
            pass

    score = max(0.0, min(score, 1.0))
    return round(score, 3), info


def score_tag(score: float, info: dict) -> str:
    """与 v1 一致。"""
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
