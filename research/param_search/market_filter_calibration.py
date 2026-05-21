"""
research/param_search/market_filter_calibration.py — 任务3：market_filter 阈值校准

data/market_filter.py 当前 6 个阈值都没有回测依据：
  第一层（当日广度）:
    UP_RATIO_MIN        = 0.45    上涨家数占比最低门槛
    UP_RATIO_WEAK       = 0.40    宽松门槛（配合中位数）
    BIG_DROP_MAX        = 0.20    大跌股（<-4%）占比上限
    MEDIAN_PCT_MIN      = -0.5    中位涨跌幅下限 %
  第二层（10 日趋势）:
    TREND_UP_RATIO_MIN  = 0.42    过去 10 日均上涨占比
    TREND_MEDIAN_MIN    = -0.3    过去 10 日均中位涨跌幅

  当前规则 = (up>=45 且 big_drop<=20)  或  (up>=40 且 median>=-0.5)
            AND
            (10日均 up>=42 AND 10日均 median>=-0.3)
  约 40% 交易日被过滤掉。问题：这套阈值过滤的方向对吗？阈值在最优值附近吗？

测法：
  1. 把过去 27 个月（IS+OOS）每个交易日的 6 个广度指标算出来
  2. 把这些日子上发出的所有共振信号 + ret_v4 join 上去
  3. 比较：通过过滤 vs 被过滤掉那批日子，发出的信号实际 V4 收益如何？
  4. 6 个阈值各做敏感性分析：threshold 取 [低, 中, 高] 看 retained 信号均收益变化
  5. 给出数据驱动的建议阈值

⚠️ caveat：
  - 真正的"过滤效果"应该看"被过滤掉那天产出的信号 vs 通过那天产出的信号"
    我用 in-memory 扫描在所有交易日（不论过滤与否）都生成信号，然后看 day-level 的统计
  - 一个被过滤掉的日子，scanner 在生产里是不会输出信号的，但本质上信号已经存在
    用"被过滤的日子的潜在信号"对比"通过的日子的实际信号"是 honest 的反事实比较
"""

from __future__ import annotations

import os
import sys
import time
import sqlite3

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import DB_PATH
from research.param_search import _common as C
from research.param_search import resonance_exit_compare as REC

IN_START,  IN_END  = "2024-01-01", "2025-09-30"
OOS_START, OOS_END = "2025-10-01", "2026-03-31"

# 当前阈值（来自 data/market_filter.py）
CUR = {
    "UP_RATIO_MIN":       0.45,
    "UP_RATIO_WEAK":      0.40,
    "BIG_DROP_MAX":       0.20,
    "MEDIAN_PCT_MIN":     -0.5,
    "TREND_UP_RATIO_MIN": 0.42,
    "TREND_MEDIAN_MIN":   -0.3,
    "TREND_WINDOW":       10,
}


def _ret_v4(p, sig_idx):
    n = p["n"]; buy_idx = sig_idx + 1
    if buy_idx >= n: return None
    bp = p["open"][buy_idx]
    if bp <= 0: return None
    target, stop = bp * 1.10, bp * 0.95
    last_idx = min(buy_idx + 20 - 1, n - 1)
    for k in range(buy_idx, last_idx + 1):
        if p["low"][k] <= stop:  return float((stop - bp) / bp * 100.0)
        if p["high"][k] >= target: return float((target - bp) / bp * 100.0)
        if k == last_idx: return float((p["close"][k] - bp) / bp * 100.0)
    return None


def build_dataset(start, end, label):
    """
    返回两个 DataFrame:
      daily_df : 每个交易日的 6 个广度指标 + 当日信号统计
      signal_df: 每条信号 + 当日所在的 6 个广度指标
    """
    print(f"\n[{label}] 加载行情 ...", flush=True); t0 = time.time()
    raw, name_map, _ = C.load_universe(start, end)
    prep = C.prepare_universe(raw, name_map)
    print(f"  股票池 {len(prep)} 只  {time.time()-t0:.1f}s")
    del raw

    # ── 全交易日的 day-level 广度 ──
    trade_dates = REC._all_trade_dates(DB_PATH, start, end)
    breadth = REC._build_breadth(prep, trade_dates)

    daily_rows = []
    for td in trade_dates:
        b = breadth.get(td)
        if b is None: continue
        daily_rows.append({"trade_date": td,
                           "up_ratio":  b["up_ratio"],          # 0~1
                           "big_drop":  b["big_drop"],          # 0~1
                           "median":    b["median"],            # %
                           })
    daily = pd.DataFrame(daily_rows).sort_values("trade_date").reset_index(drop=True)

    # 10 日趋势
    daily["trend_up"]     = daily["up_ratio"].rolling(CUR["TREND_WINDOW"], min_periods=5).mean()
    daily["trend_median"] = daily["median"].rolling(CUR["TREND_WINDOW"], min_periods=5).mean()

    # 当前规则下"过滤通过"标记
    def _pass(r):
        if pd.isna(r["trend_up"]) or pd.isna(r["trend_median"]):
            day_ok = ((r["up_ratio"] >= CUR["UP_RATIO_MIN"] and r["big_drop"] <= CUR["BIG_DROP_MAX"])
                      or (r["up_ratio"] >= CUR["UP_RATIO_WEAK"] and r["median"] >= CUR["MEDIAN_PCT_MIN"]))
            return day_ok
        day_ok = ((r["up_ratio"] >= CUR["UP_RATIO_MIN"] and r["big_drop"] <= CUR["BIG_DROP_MAX"])
                  or (r["up_ratio"] >= CUR["UP_RATIO_WEAK"] and r["median"] >= CUR["MEDIAN_PCT_MIN"]))
        trend_ok = (r["trend_up"] >= CUR["TREND_UP_RATIO_MIN"]
                    and r["trend_median"] >= CUR["TREND_MEDIAN_MIN"])
        return day_ok and trend_ok
    daily["cur_pass"] = daily.apply(_pass, axis=1)

    # ── 共振信号（用 in-memory 扫描，**所有日子都扫**，不应用 market filter） ──
    # REC._scan_resonance 内部要 market filter；为了拿"反事实信号"，自己造一个全 True breadth
    REC._precompute_signals(prep)
    breadth_all_ok = {td: {"up_ratio": 1.0, "big_drop": 0.0, "median": 1.0} for td in trade_dates}
    sigs = REC._scan_resonance(prep, breadth_all_ok, trade_dates, start, end)
    print(f"  全扫不过滤信号 {len(sigs)} 条")

    # 给每条信号算 ret_v4 + join 当日 breadth
    daily_map = {r["trade_date"]: r for _, r in daily.iterrows()}
    rows = []
    for s in sigs:
        sym = s["symbol"]; p = prep[sym]; sig_idx = s["sig_idx"]
        rv = _ret_v4(p, sig_idx)
        if rv is None: continue
        td = s["signal_date"]
        b = daily_map.get(td)
        if b is None: continue
        rows.append({"symbol": sym, "signal_date": td, "ret_v4": rv,
                     "up_ratio": b["up_ratio"], "big_drop": b["big_drop"], "median": b["median"],
                     "trend_up": b["trend_up"], "trend_median": b["trend_median"],
                     "cur_pass": bool(b["cur_pass"])})
    signals = pd.DataFrame(rows)
    print(f"  最终样本 {len(signals)} 条")

    # 给 daily 加上"当日产了多少信号 + 平均 ret_v4"统计
    by_date = signals.groupby("signal_date").agg(n_sigs=("ret_v4","size"),
                                                  avg_ret_v4=("ret_v4","mean")).reset_index()
    daily = daily.merge(by_date, left_on="trade_date", right_on="signal_date", how="left")
    daily["n_sigs"] = daily["n_sigs"].fillna(0).astype(int)
    return daily, signals


# ════════════════════════════════════════════════════════════
#  分析
# ════════════════════════════════════════════════════════════

def overall_filter_effect(signals: pd.DataFrame, label: str):
    print(f"\n  [{label}] 当前 market_filter 在共振信号上的效果")
    pass_sigs = signals[signals["cur_pass"]]
    drop_sigs = signals[~signals["cur_pass"]]
    n_p, n_d = len(pass_sigs), len(drop_sigs)
    print(f"    通过过滤  n={n_p:>5}  ret_v4 均 {pass_sigs['ret_v4'].mean():>+5.2f}%  胜率 {(pass_sigs['ret_v4']>0).mean()*100:>4.1f}%  中位 {pass_sigs['ret_v4'].median():>+5.2f}%")
    print(f"    被过滤   n={n_d:>5}  ret_v4 均 {drop_sigs['ret_v4'].mean():>+5.2f}%  胜率 {(drop_sigs['ret_v4']>0).mean()*100:>4.1f}%  中位 {drop_sigs['ret_v4'].median():>+5.2f}%")
    if n_d > 0:
        print(f"    Δ均收益（通过 - 被滤）: {pass_sigs['ret_v4'].mean()-drop_sigs['ret_v4'].mean():+.2f}%   "
              f"Δ胜率: {(pass_sigs['ret_v4']>0).mean()*100 - (drop_sigs['ret_v4']>0).mean()*100:+.1f}pp")


def threshold_sweep(signals: pd.DataFrame, col: str, label: str, direction: str,
                    bins: list[float]):
    """
    direction='ge'：通过 = 当日 col >= threshold
    direction='le'：通过 = 当日 col <= threshold
    """
    print(f"\n  [{label}] 单维敏感性扫描：{col}（{direction}）")
    print(f"    {'thresh':>8} {'n_pass':>7} {'n_drop':>7} {'pass_ret':>9} {'drop_ret':>9} {'pass_win%':>9} {'Δret':>7}")
    for t in bins:
        if direction == "ge":
            mask = signals[col] >= t
        else:
            mask = signals[col] <= t
        ps = signals[mask]; ds = signals[~mask]
        if len(ps) == 0 or len(ds) == 0:
            print(f"    {t:>8.3f} {len(ps):>7} {len(ds):>7}     —          —          —      —")
            continue
        d_ret = ps["ret_v4"].mean() - ds["ret_v4"].mean()
        print(f"    {t:>8.3f} {len(ps):>7} {len(ds):>7} {ps['ret_v4'].mean():>+8.2f}% {ds['ret_v4'].mean():>+8.2f}% {(ps['ret_v4']>0).mean()*100:>8.1f}% {d_ret:>+6.2f}")


def decile_analysis(signals: pd.DataFrame, col: str, label: str):
    """把信号按 col 分成 10 等份，看每份的 ret_v4 表现，看 col 是否有 monotonic lift。"""
    print(f"\n  [{label}] {col} 十分位分析")
    sd = signals.dropna(subset=[col]).copy()
    sd["dec"] = pd.qcut(sd[col], 10, labels=False, duplicates="drop")
    g = sd.groupby("dec").agg(n=("ret_v4","size"),
                              mean=("ret_v4","mean"),
                              win=("ret_v4", lambda x: (x>0).mean()*100),
                              q_lo=(col,"min"), q_hi=(col,"max")).round(3)
    print(f"    {'dec':>4} {'range':>22} {'n':>5} {'ret_v4 mean':>12} {'win%':>6}")
    for d, r in g.iterrows():
        print(f"    {int(d):>4} [{r['q_lo']:>+7.3f}, {r['q_hi']:>+7.3f}] {int(r['n']):>5} {r['mean']:>+11.2f}% {r['win']:>5.1f}%")


def main():
    print(f"\n{'#'*70}")
    print(f"#  任务3 · market_filter 阈值校准")
    print(f"#  IS  {IN_START}~{IN_END}    OOS {OOS_START}~{OOS_END}")
    print(f"#  当前阈值: {CUR}")
    print(f"{'#'*70}")

    daily_is,  sig_is  = build_dataset(IN_START,  IN_END,  "IS")
    daily_oos, sig_oos = build_dataset(OOS_START, OOS_END, "OOS")

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
    sig_is.to_csv(os.path.join(out_dir, "mfilter_signals_is.csv"),  index=False, encoding="utf-8-sig")
    sig_oos.to_csv(os.path.join(out_dir, "mfilter_signals_oos.csv"), index=False, encoding="utf-8-sig")

    # ── Q1: 当前过滤的整体效果 ──
    print(f"\n{'='*70}")
    print(f"  Q1: 当前 filter 在共振信号上是否真的把'坏信号'过滤掉了？")
    print(f"{'='*70}")
    overall_filter_effect(sig_is,  "IS")
    overall_filter_effect(sig_oos, "OOS")

    # ── Q2: 每个阈值单独扫描 ──
    print(f"\n{'='*70}")
    print(f"  Q2: 6 个阈值各自的敏感性扫描（其他维度不动）")
    print(f"{'='*70}")
    for label, sigs in (("IS", sig_is), ("OOS", sig_oos)):
        threshold_sweep(sigs, "up_ratio",   label, "ge", [0.30, 0.35, 0.40, 0.45, 0.50, 0.55])
        threshold_sweep(sigs, "big_drop",   label, "le", [0.10, 0.15, 0.20, 0.25, 0.30])
        threshold_sweep(sigs, "median",     label, "ge", [-1.0, -0.5, 0.0, 0.3, 0.5])
        threshold_sweep(sigs, "trend_up",   label, "ge", [0.35, 0.40, 0.42, 0.45, 0.50])
        threshold_sweep(sigs, "trend_median", label, "ge", [-0.5, -0.3, 0.0, 0.2, 0.4])

    # ── Q3: 十分位分析（看 monotonic）──
    print(f"\n{'='*70}")
    print(f"  Q3: 每个指标的十分位分析（看有无 monotonic relationship）")
    print(f"{'='*70}")
    for label, sigs in (("IS", sig_is), ("OOS", sig_oos)):
        for c in ("up_ratio", "big_drop", "median", "trend_up", "trend_median"):
            decile_analysis(sigs, c, label)

    # ── Q4: 当前过滤拒掉的那批日子的"反事实信号"详查 ──
    print(f"\n{'='*70}")
    print(f"  Q4: 被现 filter 拒掉的那批信号到底是什么样的")
    print(f"{'='*70}")
    for label, sigs in (("IS", sig_is), ("OOS", sig_oos)):
        ds = sigs[~sigs["cur_pass"]]
        if len(ds) == 0:
            print(f"\n  [{label}] 被过滤的信号数: 0"); continue
        print(f"\n  [{label}] 被过滤 n={len(ds)}，ret_v4 均{ds['ret_v4'].mean():+.2f}% 胜率{(ds['ret_v4']>0).mean()*100:.1f}%")
        # 按 ret_v4 分布看
        worst = ds.nsmallest(5, "ret_v4")
        best = ds.nlargest(5, "ret_v4")
        print(f"    被滤里最差 5 条：{[(r['signal_date'], r['symbol'], r['ret_v4']) for _, r in worst.iterrows()]}")
        print(f"    被滤里最好 5 条：{[(r['signal_date'], r['symbol'], r['ret_v4']) for _, r in best.iterrows()]}")


if __name__ == "__main__":
    t = time.time()
    main()
    print(f"\n总耗时 {time.time()-t:.1f}s")
