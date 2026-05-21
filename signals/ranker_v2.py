"""
signals/ranker_v2.py — 数据驱动的 ranker 重写（任务2 落地）

跟 ranker.py 并存，**不动原版**。如果上线要切，scanner.py 改成 import ranker_v2 就行。

为什么改：
  research/param_search/ranker_calibration.py 跑了 1869 条历史共振信号（IS+OOS）：
    - 原 ranker 5 维总分 R² < 2%，桶之间没干净 lift
    - 信号新鲜度（25% 权重）在共振池里**结构性失效**：scanner 只 pick 当日 fresh 命中，
      所以"today_hit"几乎恒等于 1，这 25% 权重没东西可调
    - 共振强度（20%）：范围 0.13~0.20 太窄，几乎无变量
    - 趋势对齐（20%）：IS 显著 / OOS 不显著，regime-dependent
    - 流动性（15%）：数据不支持，IS/OOS 符号反向
    - 风险扣分（20%）：其中"距 120 日高位 ≥ 0.97"是唯一在 IS+OOS 都稳的负预测因子

设计思路：
  1. 硬过滤层（直接返回 0）：ST、距 120 日高位 ≥ 0.97
     —— 把"避开追高"从 -0.05 软扣分升级为硬门（数据支持的最强信号）
  2. 软评分 base=0.5，加减都受限制；权重按"数据支持的程度"分配
     —— 信号新鲜度 0%（结构失效）；共振强度 5%（聊胜于无）；趋势对齐 10%；
        流动性 5%；风险/位置调整 ±0.3
  3. 评分范围 [0, 1]，与原版接口完全一致；info dict 多了 "rejected" 字段
"""

from __future__ import annotations
import math
import pandas as pd


def calc_signal_score(
    df: pd.DataFrame,
    hit_strategies: set | list,
    hit_dates: set | list,
    name: str,
    latest_date: str,
) -> tuple[float, dict]:
    """
    接口与 signals/ranker.py.calc_signal_score 一致，返回 (score, info)。
    info 多了 "rejected" 字段：被硬过滤时记原因，否则为 None。
    """
    info = {"today_hit": True, "high_position": False, "gain_60d": 0.0,
            "rejected": None, "version": "v2"}

    if df is None or df.empty or len(df) < 20:
        return 0.0, info

    close = df["close"].astype(float)

    # ── 硬过滤 1：ST ──
    if name and "ST" in str(name).upper().replace(" ", ""):
        info["rejected"] = "ST"
        info["high_position"] = False
        return 0.0, info

    # ── 硬过滤 2：距 120 日高位 ≥ 0.97 ──
    if len(close) >= 30:
        try:
            window_high = float(close.tail(120).max())
            cur = float(close.iloc[-1])
            ratio_to_high = cur / window_high if window_high > 0 else 0
            if ratio_to_high >= 0.97:
                info["rejected"] = "距120日高位≥97%"
                info["high_position"] = True
                return 0.0, info
        except Exception:
            ratio_to_high = 0
    else:
        ratio_to_high = 0

    # ── 软评分 (base=0.5) ──
    score = 0.5

    # 共振强度 ±0.05（n_strats: 2→0, 3→+0.05, 4+→+0.05）
    n_strats = len(hit_strategies) if hit_strategies else 0
    score += max(0.0, min((n_strats - 2) * 0.05, 0.05))

    # 趋势对齐 +0.10 max（IS 显著 / OOS 弱，给中等权重不冒进）
    if len(close) >= 60:
        ma20 = close.rolling(20).mean().iloc[-1]
        ma60 = close.rolling(60).mean().iloc[-1]
        if pd.notna(ma20) and pd.notna(ma60):
            if ma20 > ma60:
                score += 0.06
            if close.iloc[-1] > ma20:
                score += 0.04

    # 流动性 +0.05 max（数据弱支持，做温和的范围检查）
    if "amount" in df.columns and len(df) >= 20:
        amt_avg20 = df["amount"].astype(float).tail(20).mean()
        if pd.notna(amt_avg20) and amt_avg20 > 0:
            log_amt = math.log10(amt_avg20)
            dist = abs(log_amt - 8.18)
            if dist < 0.5:
                score += 0.05 * (1.0 - dist / 0.5)

    # ── 60 日涨幅：双向调整 ──
    # 高位扣分（IS 显著，OOS 弱，但落在硬过滤之外的范围里做软扣）
    # 低位加分（贴合 boll_rv 均值回归哲学，IS 显著）
    if len(close) >= 60:
        try:
            base = float(close.iloc[-60])
            if base > 0:
                gain_60d = (float(close.iloc[-1]) / base - 1) * 100
                info["gain_60d"] = round(gain_60d, 1)
                if gain_60d > 50:
                    score -= 0.20
                    info["high_position"] = True
                elif gain_60d > 30:
                    score -= 0.10
                    info["high_position"] = True
                elif gain_60d < -25:
                    score += 0.15
                elif gain_60d < -10:
                    score += 0.08
        except Exception:
            pass

    # ── 距 120 日高位的软扣分（已通过 0.97 硬过滤，这里管 0.85~0.97）──
    if ratio_to_high >= 0.92:
        score -= 0.10
        info["high_position"] = True
    elif ratio_to_high >= 0.85:
        score -= 0.05
        info["high_position"] = True

    score = max(0.0, min(score, 1.0))
    return round(score, 3), info


def score_tag(score: float, info: dict) -> str:
    """与原版相同，但增加硬过滤的明确提示。"""
    if info.get("rejected"):
        return f"❌ 已过滤({info['rejected']})"
    if score >= 0.70:
        return "🔥 强推"
    if score < 0.45:
        return "❌ 不建议"
    if info.get("high_position", False):
        return "⚠️ 注意高位"
    if score >= 0.60:
        return "👌 可入"
    return ""
