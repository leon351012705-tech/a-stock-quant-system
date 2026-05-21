"""
research/param_search/eval_mfilter_v2.py — 用历史共振信号对比 v1 / v2 market_filter

数据来自 market_filter_calibration.py 留下的 mfilter_signals_*.csv。
"""

from __future__ import annotations
import os
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
_HERE = os.path.dirname(os.path.abspath(__file__))

# ── 两套阈值 ──
V1 = {"UP_MIN": 0.45, "UP_WEAK": 0.40, "BIG_DROP_MAX": 0.20, "MED_MIN": -0.5,
      "T_UP_MIN": 0.42, "T_MED_MIN": -0.3}
V2 = {"UP_MIN": 0.40, "BIG_DROP_MAX": None, "MED_MIN": -0.5,
      "T_UP_MIN": 0.45, "T_MED_MIN": -0.3}


def v1_pass(r):
    if pd.isna(r.get("trend_up")) or pd.isna(r.get("trend_median")):
        return ((r["up_ratio"] >= V1["UP_MIN"] and r["big_drop"] <= V1["BIG_DROP_MAX"])
                or (r["up_ratio"] >= V1["UP_WEAK"] and r["median"] >= V1["MED_MIN"]))
    day = ((r["up_ratio"] >= V1["UP_MIN"] and r["big_drop"] <= V1["BIG_DROP_MAX"])
           or (r["up_ratio"] >= V1["UP_WEAK"] and r["median"] >= V1["MED_MIN"]))
    trend = (r["trend_up"] >= V1["T_UP_MIN"] and r["trend_median"] >= V1["T_MED_MIN"])
    return day and trend


def v2_pass(r):
    if pd.isna(r.get("trend_up")) or pd.isna(r.get("trend_median")):
        return (r["up_ratio"] >= V2["UP_MIN"] and r["median"] >= V2["MED_MIN"])
    day = (r["up_ratio"] >= V2["UP_MIN"] and r["median"] >= V2["MED_MIN"])
    trend = (r["trend_up"] >= V2["T_UP_MIN"] and r["trend_median"] >= V2["T_MED_MIN"])
    return day and trend


def report(df, label):
    df = df.copy()
    df["v1_pass"] = df.apply(v1_pass, axis=1)
    df["v2_pass"] = df.apply(v2_pass, axis=1)
    print(f"\n[{label}]  n={len(df)} 条历史共振信号")
    for name, mask in (("v1 通过", df["v1_pass"]), ("v1 拒掉", ~df["v1_pass"]),
                       ("v2 通过", df["v2_pass"]), ("v2 拒掉", ~df["v2_pass"])):
        sub = df[mask]
        if len(sub) == 0:
            print(f"  {name:<10} n=0"); continue
        print(f"  {name:<10} n={len(sub):>5}  ret_v4 均 {sub['ret_v4'].mean():>+5.2f}%  "
              f"中位 {sub['ret_v4'].median():>+5.2f}%  胜率 {(sub['ret_v4']>0).mean()*100:>4.1f}%")

    # 详细：v1 vs v2 的 4 个象限（pass 一致/分歧）
    print(f"\n  ── v1 / v2 决策一致性（4 个象限）──")
    for v1_flag in (True, False):
        for v2_flag in (True, False):
            sub = df[(df["v1_pass"] == v1_flag) & (df["v2_pass"] == v2_flag)]
            if len(sub) == 0:
                continue
            v1lab = "通过" if v1_flag else "拒掉"; v2lab = "通过" if v2_flag else "拒掉"
            label_tag = f"v1{v1lab}/v2{v2lab}"
            print(f"    {label_tag:<14} n={len(sub):>5}  ret_v4 均 {sub['ret_v4'].mean():>+5.2f}%  "
                  f"胜率 {(sub['ret_v4']>0).mean()*100:>4.1f}%")

    # 谁是赢家：通过组的均收益差
    v1_pass_ret = df[df["v1_pass"]]["ret_v4"].mean()
    v2_pass_ret = df[df["v2_pass"]]["ret_v4"].mean()
    v1_pass_win = (df[df["v1_pass"]]["ret_v4"] > 0).mean() * 100
    v2_pass_win = (df[df["v2_pass"]]["ret_v4"] > 0).mean() * 100
    v1_n = df["v1_pass"].sum()
    v2_n = df["v2_pass"].sum()
    print(f"\n  ── 总结 [{label}] ──")
    print(f"    v1 通过 {v1_n} 条（{v1_n/len(df)*100:.0f}%），均收益 {v1_pass_ret:+.2f}%，胜率 {v1_pass_win:.1f}%")
    print(f"    v2 通过 {v2_n} 条（{v2_n/len(df)*100:.0f}%），均收益 {v2_pass_ret:+.2f}%，胜率 {v2_pass_win:.1f}%")
    print(f"    Δ通过数 {v2_n - v1_n:+d}（{(v2_n-v1_n)/len(df)*100:+.0f}pp）  "
          f"Δ通过组均收益 {v2_pass_ret - v1_pass_ret:+.2f}pp  "
          f"Δ通过组胜率 {v2_pass_win - v1_pass_win:+.1f}pp")


def main():
    out_dir = os.path.join(_HERE, "out")
    df_is  = pd.read_csv(os.path.join(out_dir, "mfilter_signals_is.csv"))
    df_oos = pd.read_csv(os.path.join(out_dir, "mfilter_signals_oos.csv"))
    print(f"\n{'#'*60}")
    print(f"#  market_filter v1 vs v2 对比")
    print(f"{'#'*60}")
    report(df_is,  "IS  2024-01~2025-09")
    report(df_oos, "OOS 2025-10~2026-03")


if __name__ == "__main__":
    main()
