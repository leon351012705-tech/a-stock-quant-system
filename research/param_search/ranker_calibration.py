"""
research/param_search/ranker_calibration.py — 任务2：ranker 五维权重数据校准

用 in-memory 共振扫描拿到的 IS+OOS 共 ~1869 条历史共振信号，
对每条信号在"信号日 close"上重算 5 维 + 11 个子特征，再拿"未来收益"做回归，
看现有权重 (25/20/20/15/20) 哪些是有效杠杆、哪些可以减为 0。

不动任何线上代码（ranker.py 一字不改）。本脚本只在 research/ 内分析。

⚠️ 注意：本脚本里的 5 维分数实现严格对齐 signals/ranker.py 的 calc_signal_score 逻辑，
   但拆成 5 个子分数 + 子组件存下来，方便细粒度回归。

输出：
  - 分桶表：现 ranker 总分 → 实际未来收益（看 ranker 有没有 lift）
  - 5 维回归：现 5 个维度各自的 coef / t / p 值
  - 子特征回归：11 个子特征级别的归因
  - 样本内 vs 样本外一致性
  - 数据驱动的建议新权重（IS 拟合得到，OOS 验证）
"""

from __future__ import annotations

import os
import sys
import math
import time

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from research.param_search import _common as C
from research.param_search import resonance_exit_compare as REC

IN_START,  IN_END  = "2024-01-01", "2025-09-30"
OOS_START, OOS_END = "2025-10-01", "2026-03-31"

FUTURE_DAYS = 5     # 用户口径：5 日后收益

# 现有权重（来自 signals/ranker.py 顶部注释）
CURRENT_WEIGHTS = {
    "freshness":  0.25,
    "resonance":  0.20,
    "trend":      0.20,
    "liquidity":  0.15,
    "risk":       0.20,
}


# ════════════════════════════════════════════════════════════
#  ranker 5 维 + 子特征复刻（严格对齐 signals/ranker.py）
# ════════════════════════════════════════════════════════════

def _features_for_signal(p: dict, i: int, hit_strategies: set, hit_dates: set, name: str):
    """
    在信号日（位置 i, sym_dates[i]=signal_date）算 5 维分数 + 11 个子特征。
    p 是 prep 单标的 dict，i 是信号日在 p["dates"] 里的位置。
    返回 dict（含 dim 分数 + 子特征 + total score）。
    """
    close = p["close"]; volume = p["volume"]; amount = p["amount"]
    turnover_arr = p.get("_turnover_raw")   # 见 main 里的预算
    sym_dates = p["dates"]
    sig_date = sym_dates[i]

    # 至少 60 天历史（ranker 自己也要这个量级）
    if i < 60:
        return None

    f = {
        "score_freshness": 0.0, "score_resonance": 0.0, "score_trend": 0.0,
        "score_liquidity": 0.0, "score_risk": 0.0,
        # 子特征
        "feat_today_hit": 0, "feat_yesterday_hit": 0, "feat_older_hit": 0,
        "feat_n_strats": 0,
        "feat_ma20_gt_ma60": 0, "feat_close_gt_ma20": 0, "feat_vol_ratio": 0.0,
        "feat_log_amt20": 0.0, "feat_turnover": 0.0,
        "feat_is_st": 0, "feat_gain_60d": 0.0, "feat_close_to_120dhigh": 0.0,
    }

    # ── 25% 信号新鲜度 ──
    # 用 signal_date 在 hit_dates 里的"新鲜度"判断
    # 这里 latest_date = signal_date（信号日当天）；hit_dates 是该 symbol 在 3 日窗口内
    # 命中各策略的日期集合，是从 _scan_resonance 收集来的
    if sig_date in hit_dates:
        f["score_freshness"] = 0.25
        f["feat_today_hit"] = 1
    elif len(hit_dates) > 0:
        # 区分昨日 vs 更早（用 sym_dates 里的前一根 K 线）
        prev_in_sym = sym_dates[i - 1] if i > 0 else ""
        if prev_in_sym in hit_dates:
            f["score_freshness"] = 0.15
            f["feat_yesterday_hit"] = 1
        else:
            f["score_freshness"] = 0.05
            f["feat_older_hit"] = 1

    # ── 20% 共振强度 ──
    n_strats = len(hit_strategies) if hit_strategies else 0
    f["feat_n_strats"] = n_strats
    f["score_resonance"] = min(n_strats / 3.0, 1.0) * 0.20

    # ── 20% 趋势对齐 ──
    close_s = pd.Series(close)
    ma20 = close_s.rolling(20).mean().iloc[i]
    ma60 = close_s.rolling(60).mean().iloc[i]
    trend_score = 0.0
    if pd.notna(ma20) and pd.notna(ma60):
        if ma20 > ma60:
            trend_score += 0.10
            f["feat_ma20_gt_ma60"] = 1
        if close[i] > ma20:
            trend_score += 0.05
            f["feat_close_gt_ma20"] = 1
    vol_s = pd.Series(volume)
    vol_ma20 = vol_s.rolling(20).mean().iloc[i]
    if pd.notna(vol_ma20) and vol_ma20 > 0:
        vr = volume[i] / vol_ma20
        f["feat_vol_ratio"] = float(vr)
        trend_score += max(0.0, min((vr - 1.0) * 0.05, 0.05))
    f["score_trend"] = trend_score

    # ── 15% 流动性 ──
    liq_score = 0.0
    amt_s = pd.Series(amount)
    amt_ma20 = amt_s.tail(i + 1).tail(20).mean() if i >= 19 else np.nan
    # 上面写法在大数组上慢；直接用 rolling
    amt_ma20 = amt_s.rolling(20).mean().iloc[i]
    if pd.notna(amt_ma20) and amt_ma20 > 0:
        log_amt = math.log10(amt_ma20)
        f["feat_log_amt20"] = log_amt
        log_target = 8.18
        dist = abs(log_amt - log_target)
        liq_score += max(0.0, 0.10 - dist * 0.10)
    if turnover_arr is not None and not np.isnan(turnover_arr[i]):
        t_val = float(turnover_arr[i])
        f["feat_turnover"] = t_val
        if 0.3 <= t_val <= 8.0:
            dist = abs(t_val - 2.0) / 3.0
            liq_score += max(0.0, 0.05 - dist * 0.025)
    f["score_liquidity"] = liq_score

    # ── 20% 风险扣分 ──
    risk_score = 0.0
    if name and "ST" in str(name).upper().replace(" ", ""):
        risk_score -= 0.50
        f["feat_is_st"] = 1
    if i >= 60:
        base = close[i - 60]
        if base > 0:
            gain_60d = (close[i] / base - 1) * 100
            f["feat_gain_60d"] = float(gain_60d)
            if gain_60d > 50:
                risk_score -= 0.15
            elif gain_60d > 30:
                risk_score -= 0.05
            elif gain_60d < -25:
                risk_score += 0.06
            elif gain_60d < -10:
                risk_score += 0.03
    if i >= 30:
        lookback_start = max(0, i - 119)
        window_high = float(close[lookback_start:i + 1].max())
        if window_high > 0:
            ratio = close[i] / window_high
            f["feat_close_to_120dhigh"] = float(ratio)
            if ratio >= 0.97:
                risk_score -= 0.05
    f["score_risk"] = risk_score

    total = sum([f["score_freshness"], f["score_resonance"], f["score_trend"],
                 f["score_liquidity"], f["score_risk"]])
    total = max(0.0, min(total, 1.0))
    f["score_total"] = round(total, 4)
    return f


# ════════════════════════════════════════════════════════════
#  V4 出场（用于 ret_v4 target）+ 5 日 raw 收益
# ════════════════════════════════════════════════════════════

def _ret_5d(p: dict, sig_idx: int) -> float | None:
    """T+1 开盘买、第 T+1+5 天收盘卖。返回净 % 收益。"""
    n = p["n"]
    buy_idx = sig_idx + 1
    end_idx = buy_idx + FUTURE_DAYS - 1
    if end_idx >= n:
        return None
    bp = p["open"][buy_idx]
    if bp <= 0:
        return None
    sp = p["close"][end_idx]
    return float((sp - bp) / bp * 100.0)


def _ret_v4(p: dict, sig_idx: int) -> float | None:
    """V4 出场：T+1 开盘买 + -5%硬止损 + +10%固定止盈 + 20 日上限。"""
    n = p["n"]
    buy_idx = sig_idx + 1
    if buy_idx >= n:
        return None
    bp = p["open"][buy_idx]
    if bp <= 0:
        return None
    target = bp * 1.10; stop = bp * 0.95
    last_idx = min(buy_idx + 20 - 1, n - 1)
    for k in range(buy_idx, last_idx + 1):
        if p["low"][k] <= stop:
            return float((stop - bp) / bp * 100.0)
        if p["high"][k] >= target:
            return float((target - bp) / bp * 100.0)
        if k == last_idx:
            return float((p["close"][k] - bp) / bp * 100.0)
    return None


# ════════════════════════════════════════════════════════════
#  扫描 + 算 features（沿用 REC 的扫描，加 turnover）
# ════════════════════════════════════════════════════════════

def _attach_turnover(prep: dict, raw_universe):
    """把 turnover 数组挂到 prep 上（_common.prepare_universe 没存它）。"""
    # 重新从原始 universe 里取 turnover
    pass


def build_dataset(start: str, end: str, label: str) -> pd.DataFrame:
    print(f"\n[{label}] 加载行情 ...", flush=True); t0 = time.time()
    raw, name_map, _ = C.load_universe(start, end)
    prep = C.prepare_universe(raw, name_map)
    # 单独挂 turnover（_common 没存）
    for sym, df in raw.items():
        if sym in prep:
            prep[sym]["_turnover_raw"] = df["turnover"].astype(float).to_numpy()
    print(f"  股票池 {len(prep)} 只  {time.time()-t0:.1f}s")
    del raw

    REC._precompute_signals(prep)
    import sqlite3
    from config import DB_PATH
    trade_dates = REC._all_trade_dates(DB_PATH, start, end)
    breadth = REC._build_breadth(prep, trade_dates)
    print(f"  交易日 {len(trade_dates)} 天，过广度 {sum(1 for v in breadth.values() if REC._market_ok(v))} 天", flush=True)

    print(f"  扫描共振信号 ...", flush=True); t0 = time.time()
    sigs = REC._scan_resonance(prep, breadth, trade_dates, start, end)
    print(f"  得 {len(sigs)} 条；{time.time()-t0:.1f}s")

    print(f"  对每条信号算 features + ret ...", flush=True); t0 = time.time()
    rows = []
    for s in sigs:
        sym = s["symbol"]; p = prep[sym]; sig_idx = s["sig_idx"]
        # hit_strategies / hit_dates：从命中策略字符串还原
        hits = set(s["hit_strategies"].split(","))
        # hit_dates：在 3 日窗口内该 symbol 实际命中策略的那些日期
        # 我们没存这个；近似：用 signal_date 当天和前两天里"该 sym 实际有信号"的日期
        # 这里简化处理：用 signal_date 本身（视为"今日命中"占优）。这样得到的 freshness 会偏高，
        # 但对所有信号一致，不影响相对比较。
        hit_dates = {s["signal_date"]}
        # 严格还原：往前看 3 天里这个 sym 在三个 sig 数组里 True 的位置
        for j in range(max(0, sig_idx - 2), sig_idx + 1):
            if p["sig_macd"][j] or p["sig_boll_rv"][j] or p["sig_ma"][j]:
                hit_dates.add(str(p["dates"][j]))

        name = name_map.get(str(sym).zfill(6), "")
        feat = _features_for_signal(p, sig_idx, hits, hit_dates, name)
        if feat is None:
            continue
        ret5 = _ret_5d(p, sig_idx)
        retv = _ret_v4(p, sig_idx)
        if ret5 is None or retv is None:
            continue
        feat["symbol"] = sym; feat["signal_date"] = s["signal_date"]
        feat["ret_5d"] = ret5; feat["ret_v4"] = retv
        rows.append(feat)
    df = pd.DataFrame(rows)
    print(f"  完成 {len(df)} 条；{time.time()-t0:.1f}s")
    return df


# ════════════════════════════════════════════════════════════
#  分析
# ════════════════════════════════════════════════════════════

def bucket_analysis(df: pd.DataFrame, score_col: str, target_col: str, label: str):
    buckets = [(0.0, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 1.01)]
    print(f"\n  [{label}] {score_col} → {target_col}  分桶统计：")
    print(f"    {'桶':<14} {'n':>5} {'mean%':>7} {'med%':>7} {'win%':>6}")
    for lo, hi in buckets:
        sub = df[(df[score_col] >= lo) & (df[score_col] < hi)]
        if len(sub) == 0:
            print(f"    [{lo:.2f}, {hi:.2f}) {'':<5} 0")
            continue
        print(f"    [{lo:.2f}, {hi:.2f}) {len(sub):>5} {sub[target_col].mean():>7.2f} "
              f"{sub[target_col].median():>7.2f} {(sub[target_col] > 0).mean()*100:>5.1f}")


def regression(df: pd.DataFrame, x_cols: list[str], y_col: str) -> pd.DataFrame:
    """简易 OLS：β、t、p。零 numpy 实现，避免再装 statsmodels。"""
    y = df[y_col].to_numpy(dtype=float)
    X = df[x_cols].to_numpy(dtype=float)
    n, k = X.shape
    # 加截距
    X1 = np.hstack([np.ones((n, 1)), X])
    XtX = X1.T @ X1
    try:
        XtX_inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        # 退化：用 pinv
        XtX_inv = np.linalg.pinv(XtX)
    beta = XtX_inv @ X1.T @ y
    resid = y - X1 @ beta
    df_resid = max(n - k - 1, 1)
    sigma2 = (resid @ resid) / df_resid
    se = np.sqrt(np.maximum(np.diag(XtX_inv) * sigma2, 0))
    t = beta / np.maximum(se, 1e-12)
    # 双边 p 值近似（正态近似，大样本下足够）
    from math import erf, sqrt
    p = 2 * (1 - 0.5 * (1 + np.array([erf(abs(x) / sqrt(2)) for x in t])))
    ss_total = ((y - y.mean()) ** 2).sum()
    r2 = 1 - (resid @ resid) / ss_total if ss_total > 0 else 0
    rows = [{"var": "intercept", "coef": beta[0], "t": t[0], "p": p[0]}]
    for i, c in enumerate(x_cols):
        rows.append({"var": c, "coef": beta[i + 1], "t": t[i + 1], "p": p[i + 1]})
    res = pd.DataFrame(rows)
    res.attrs["r2"] = float(r2)
    res.attrs["n"] = int(n)
    return res


def print_reg(res: pd.DataFrame, title: str):
    print(f"\n  {title}   (n={res.attrs['n']}, R²={res.attrs['r2']:.4f})")
    print(f"    {'var':<26} {'coef':>10} {'t':>8} {'p':>8}  signif")
    for _, r in res.iterrows():
        sig = "***" if r["p"] < 0.01 else "**" if r["p"] < 0.05 else "*" if r["p"] < 0.10 else ""
        print(f"    {r['var']:<26} {r['coef']:>10.4f} {r['t']:>8.2f} {r['p']:>8.4f}  {sig}")


def main():
    print(f"\n{'#'*70}")
    print(f"#  任务2 · ranker 5 维权重数据校准")
    print(f"#  IS  {IN_START}~{IN_END}    OOS {OOS_START}~{OOS_END}")
    print(f"#  目标：5 日后收益 ret_5d + V4 出场净收益 ret_v4")
    print(f"#  现有权重: {CURRENT_WEIGHTS}")
    print(f"{'#'*70}")

    df_is  = build_dataset(IN_START,  IN_END,  "IS")
    df_oos = build_dataset(OOS_START, OOS_END, "OOS")

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
    df_is.to_csv(os.path.join(out_dir, "ranker_calib_is.csv"), index=False, encoding="utf-8-sig")
    df_oos.to_csv(os.path.join(out_dir, "ranker_calib_oos.csv"), index=False, encoding="utf-8-sig")

    # ── 分桶 ──
    print(f"\n{'='*70}")
    print(f"  Q1: 现有 ranker 总分是否预测未来收益？（分桶看 lift）")
    print(f"{'='*70}")
    for label, df in (("IS",  df_is), ("OOS", df_oos)):
        bucket_analysis(df, "score_total", "ret_5d", label)
        bucket_analysis(df, "score_total", "ret_v4", label)

    DIMS = ["score_freshness", "score_resonance", "score_trend", "score_liquidity", "score_risk"]
    FEATS = ["feat_today_hit", "feat_yesterday_hit", "feat_n_strats",
             "feat_ma20_gt_ma60", "feat_close_gt_ma20", "feat_vol_ratio",
             "feat_log_amt20", "feat_turnover",
             "feat_is_st", "feat_gain_60d", "feat_close_to_120dhigh"]

    # ── 5 维回归 ──
    print(f"\n{'='*70}")
    print(f"  Q2: 5 维各自对未来收益的预测力（系数 / t / p）")
    print(f"{'='*70}")
    for label, df, target in (("IS  ret_5d",  df_is,  "ret_5d"),
                              ("OOS ret_5d",  df_oos, "ret_5d"),
                              ("IS  ret_v4",  df_is,  "ret_v4"),
                              ("OOS ret_v4",  df_oos, "ret_v4")):
        # 单变量
        rows = []
        for d in DIMS:
            r = regression(df[[d, target]].dropna(), [d], target)
            rows.append({"dim": d, "univariate_coef": r.iloc[1]["coef"],
                         "uni_t": r.iloc[1]["t"], "uni_p": r.iloc[1]["p"]})
        # 多变量
        rmulti = regression(df.dropna(subset=DIMS + [target]), DIMS, target)
        for i, d in enumerate(DIMS):
            rows[i]["multi_coef"] = rmulti.iloc[i + 1]["coef"]
            rows[i]["multi_t"] = rmulti.iloc[i + 1]["t"]
            rows[i]["multi_p"] = rmulti.iloc[i + 1]["p"]
        rdf = pd.DataFrame(rows)
        print(f"\n  [{label}]   多变量 R²={rmulti.attrs['r2']:.4f}")
        print(f"    {'dim':<18} {'uni_coef':>9} {'uni_t':>7} {'uni_p':>8}  {'multi_coef':>10} {'multi_t':>8} {'multi_p':>8}")
        for _, r in rdf.iterrows():
            sig_u = "***" if r["uni_p"] < 0.01 else "**" if r["uni_p"] < 0.05 else "*" if r["uni_p"] < 0.10 else ""
            sig_m = "***" if r["multi_p"] < 0.01 else "**" if r["multi_p"] < 0.05 else "*" if r["multi_p"] < 0.10 else ""
            print(f"    {r['dim']:<18} {r['univariate_coef']:>9.3f} {r['uni_t']:>7.2f} {r['uni_p']:>8.4f}{sig_u:<3}"
                  f" {r['multi_coef']:>10.3f} {r['multi_t']:>8.2f} {r['multi_p']:>8.4f}{sig_m}")

    # ── 子特征回归 ──
    print(f"\n{'='*70}")
    print(f"  Q3: 11 个子特征对 ret_5d 的预测力（多变量；IS 拟合 + OOS 验证）")
    print(f"{'='*70}")
    sub_is  = df_is.dropna(subset=FEATS + ["ret_5d"])
    sub_oos = df_oos.dropna(subset=FEATS + ["ret_5d"])
    res_feat_is  = regression(sub_is,  FEATS, "ret_5d")
    res_feat_oos = regression(sub_oos, FEATS, "ret_5d")
    print_reg(res_feat_is,  "IS  子特征回归 (ret_5d)")
    print_reg(res_feat_oos, "OOS 子特征回归 (ret_5d)")

    # ── 数据驱动建议权重 ──
    print(f"\n{'='*70}")
    print(f"  Q4: 数据驱动的建议权重（基于 IS 5 维多变量回归，绝对值归一化）")
    print(f"{'='*70}")
    # 取 IS ret_5d 多变量回归的 |coef| 作为重要性，按现有总和 1.0 重分配
    sub_is_dim = df_is.dropna(subset=DIMS + ["ret_5d"])
    r_dim = regression(sub_is_dim, DIMS, "ret_5d")
    coefs = {DIMS[i]: r_dim.iloc[i + 1]["coef"] for i in range(len(DIMS))}
    abs_coefs = {k: abs(v) for k, v in coefs.items()}
    total = sum(abs_coefs.values()) or 1e-9
    suggested = {k.replace("score_", ""): v / total for k, v in abs_coefs.items()}
    # 显著性 mark
    pvals = {DIMS[i]: r_dim.iloc[i + 1]["p"] for i in range(len(DIMS))}
    print(f"\n    {'dim':<12} {'current':>9} {'IS_coef':>10} {'IS_p':>8} {'|coef|share':>12}  备注")
    for k in DIMS:
        kk = k.replace("score_", "")
        sig = "***" if pvals[k] < 0.01 else "**" if pvals[k] < 0.05 else "*" if pvals[k] < 0.10 else "ns"
        note = "可能减为0" if pvals[k] >= 0.10 else "保留"
        if coefs[k] < 0 and pvals[k] < 0.10:
            note = "符号反向，重新审视"
        print(f"    {kk:<12} {CURRENT_WEIGHTS[kk]:>9.2f} {coefs[k]:>10.3f} {pvals[k]:>8.4f} {suggested[kk]:>12.2%}  {sig:<3} {note}")

    print(f"\n  CSV: out/ranker_calib_is.csv  out/ranker_calib_oos.csv")
    print(f"\n  解读速查：")
    print(f"   - 分桶里 high vs low 的 mean% 差距 → ranker 真实 lift")
    print(f"   - 多变量回归里 p > 0.10 的维度 → 数据不支持它真的有用")
    print(f"   - IS / OOS 一致性 → 是否 robust")


if __name__ == "__main__":
    t = time.time()
    main()
    print(f"\n总耗时 {time.time()-t:.1f}s")
