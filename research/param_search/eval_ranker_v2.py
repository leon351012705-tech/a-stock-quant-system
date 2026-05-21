"""
research/param_search/eval_ranker_v2.py — 任务2 落地评估：ranker_v2 vs ranker（v1）

把同一批 1869 条历史共振信号同时过 v1 / v2 两套打分，对比：
  - 桶分布是否更均匀（v1 有 60%+ 信号挤在 0.5~0.6 一个桶里）
  - 分桶的实际未来收益 lift 是否更明显
  - 硬过滤层踢掉了多少信号、被踢的那批 OOS 实际表现如何
  - 同区间整体均值收益（按 v1 / v2 各自的 top-K 选股做模拟）
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from research.param_search import _common as C
from research.param_search import resonance_exit_compare as REC
import signals.ranker     as ranker_v1
import signals.ranker_v2  as ranker_v2

IN_START,  IN_END  = "2024-01-01", "2025-09-30"
OOS_START, OOS_END = "2025-10-01", "2026-03-31"
FUTURE_DAYS = 5


def _ret_5d(p, sig_idx):
    n = p["n"]; buy_idx = sig_idx + 1; end_idx = buy_idx + FUTURE_DAYS - 1
    if end_idx >= n: return None
    bp = p["open"][buy_idx]; sp = p["close"][end_idx]
    if bp <= 0: return None
    return float((sp - bp) / bp * 100.0)


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


def build_scores(start, end, label):
    print(f"\n[{label}] 加载行情 ...", flush=True); t0 = time.time()
    raw, name_map, _ = C.load_universe(start, end)
    prep = C.prepare_universe(raw, name_map)
    print(f"  股票池 {len(prep)} 只  {time.time()-t0:.1f}s")

    REC._precompute_signals(prep)
    import sqlite3
    from config import DB_PATH
    trade_dates = REC._all_trade_dates(DB_PATH, start, end)
    breadth = REC._build_breadth(prep, trade_dates)
    sigs = REC._scan_resonance(prep, breadth, trade_dates, start, end)
    print(f"  得 {len(sigs)} 条共振信号")

    rows = []
    for s in sigs:
        sym = s["symbol"]; p = prep[sym]; sig_idx = s["sig_idx"]
        # df 给 ranker：取 sig_idx 往前 120 根 + 包含当日
        lo = max(0, sig_idx - 120 + 1); hi = sig_idx + 1
        df = pd.DataFrame({
            "trade_date": p["dates"][lo:hi],
            "open":   p["open"][lo:hi], "high": p["high"][lo:hi],
            "low":    p["low"][lo:hi],  "close": p["close"][lo:hi],
            "volume": p["volume"][lo:hi], "amount": p["amount"][lo:hi],
            "turnover": p.get("_turnover_raw", np.full(p["n"], np.nan))[lo:hi]
                        if "_turnover_raw" in p else np.nan,
        })
        # turnover 列：若没存，用 NaN
        if "_turnover_raw" not in p:
            df["turnover"] = np.nan
        else:
            df["turnover"] = p["_turnover_raw"][lo:hi]
        # 把行情中的 turnover 取出
        df = df.dropna(subset=["close"]).reset_index(drop=True)
        if len(df) < 20:
            continue

        hits = set(s["hit_strategies"].split(","))
        # hit_dates: 还原 3 日窗口内 sym 真实命中各策略的日期
        hit_dates = set()
        for j in range(max(0, sig_idx - 2), sig_idx + 1):
            if p["sig_macd"][j] or p["sig_boll_rv"][j] or p["sig_ma"][j]:
                hit_dates.add(str(p["dates"][j]))
        name = name_map.get(str(sym).zfill(6), "")

        s1, info1 = ranker_v1.calc_signal_score(df, hits, hit_dates, name, s["signal_date"])
        s2, info2 = ranker_v2.calc_signal_score(df, hits, hit_dates, name, s["signal_date"])

        ret5 = _ret_5d(p, sig_idx); retv = _ret_v4(p, sig_idx)
        if ret5 is None or retv is None:
            continue
        rows.append({"symbol": sym, "signal_date": s["signal_date"],
                     "score_v1": s1, "score_v2": s2,
                     "v2_rejected": info2.get("rejected"),
                     "ret_5d": ret5, "ret_v4": retv})
    df = pd.DataFrame(rows)
    print(f"  最终样本 {len(df)} 条；耗时 {time.time()-t0:.1f}s")
    return df


def bucket_table(df, score_col, target_col, label):
    buckets = [(0.0, 0.40), (0.40, 0.50), (0.50, 0.60), (0.60, 0.70), (0.70, 1.01)]
    print(f"\n  [{label}] {score_col} → {target_col}")
    print(f"    {'桶':<14} {'n':>5} {'占比':>6} {'mean%':>7} {'med%':>7} {'win%':>6}")
    total = len(df)
    for lo, hi in buckets:
        sub = df[(df[score_col] >= lo) & (df[score_col] < hi)]
        if len(sub) == 0:
            print(f"    [{lo:.2f}, {hi:.2f}) {'':>5} {0:>5.1f}%       —       —      —")
            continue
        print(f"    [{lo:.2f}, {hi:.2f}) {len(sub):>5} {len(sub)/total*100:>5.1f}% "
              f"{sub[target_col].mean():>7.2f} {sub[target_col].median():>7.2f} "
              f"{(sub[target_col] > 0).mean()*100:>5.1f}")


def rejected_analysis(df, label):
    rej = df[df["v2_rejected"].notna()]
    kept = df[df["v2_rejected"].isna()]
    print(f"\n  [{label}] v2 硬过滤：踢掉 {len(rej)}/{len(df)} = {len(rej)/len(df)*100:.1f}%")
    if len(rej) > 0:
        print(f"    被踢的那批：ret_5d 均 {rej['ret_5d'].mean():+.2f}%  ret_v4 均 {rej['ret_v4'].mean():+.2f}%  win {(rej['ret_v4']>0).mean()*100:.1f}%")
        by_reason = rej.groupby("v2_rejected").agg(n=("ret_v4", "size"),
                                                    ret5=("ret_5d", "mean"),
                                                    retv4=("ret_v4", "mean")).round(2)
        print(f"    按原因分布：")
        for r, row in by_reason.iterrows():
            print(f"      {r:<20} n={int(row['n']):>4}  ret_5d={row['ret5']:>+5.2f}%  ret_v4={row['retv4']:>+5.2f}%")
    if len(kept) > 0:
        print(f"    保留的那批：ret_5d 均 {kept['ret_5d'].mean():+.2f}%  ret_v4 均 {kept['ret_v4'].mean():+.2f}%  win {(kept['ret_v4']>0).mean()*100:.1f}%")


def topk_analysis(df, label, k_pct=0.30):
    """每只信号按各自 score 排序，看 top-K% 的实际表现 vs bottom-K%。"""
    n = len(df)
    k = max(1, int(n * k_pct))
    print(f"\n  [{label}] Top vs Bottom {k_pct*100:.0f}%（按 score 排序，比 v1 / v2 的选股能力）")
    for sc in ("score_v1", "score_v2"):
        sd = df.sort_values(sc, ascending=False).reset_index(drop=True)
        top = sd.head(k); bot = sd.tail(k)
        print(f"    {sc:<10}  top {k}:  ret_5d={top['ret_5d'].mean():+5.2f}%  ret_v4={top['ret_v4'].mean():+5.2f}%  win={(top['ret_v4']>0).mean()*100:.1f}%")
        print(f"    {sc:<10}  bot {k}:  ret_5d={bot['ret_5d'].mean():+5.2f}%  ret_v4={bot['ret_v4'].mean():+5.2f}%  win={(bot['ret_v4']>0).mean()*100:.1f}%")
        spread_v4 = top['ret_v4'].mean() - bot['ret_v4'].mean()
        print(f"    {sc:<10}  Top-Bot spread (ret_v4): {spread_v4:+.2f}%")


def main():
    print(f"\n{'#'*70}")
    print(f"#  ranker v1 vs v2  在 1869 条历史共振信号上的对比")
    print(f"{'#'*70}")

    df_is  = build_scores(IN_START,  IN_END,  "IS")
    df_oos = build_scores(OOS_START, OOS_END, "OOS")

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
    df_is.to_csv(os.path.join(out_dir, "ranker_v2_eval_is.csv"),   index=False, encoding="utf-8-sig")
    df_oos.to_csv(os.path.join(out_dir, "ranker_v2_eval_oos.csv"), index=False, encoding="utf-8-sig")

    # ── 分桶 ──
    print(f"\n{'='*70}\n  桶分布 + lift 对比\n{'='*70}")
    for label, df in (("IS", df_is), ("OOS", df_oos)):
        for sc in ("score_v1", "score_v2"):
            bucket_table(df, sc, "ret_v4", f"{label} {sc}")

    # ── 硬过滤分析 ──
    print(f"\n{'='*70}\n  v2 硬过滤层的命中情况\n{'='*70}")
    for label, df in (("IS", df_is), ("OOS", df_oos)):
        rejected_analysis(df, label)

    # ── Top vs Bottom 选股能力 ──
    print(f"\n{'='*70}\n  TopK 选股能力：v1 vs v2\n{'='*70}")
    for label, df in (("IS", df_is), ("OOS", df_oos)):
        topk_analysis(df, label, 0.30)
        topk_analysis(df, label, 0.10)

    # ── 整体均值对比 ──
    print(f"\n{'='*70}\n  整体均值对比（不排序，仅看硬过滤的过滤效果）\n{'='*70}")
    for label, df in (("IS", df_is), ("OOS", df_oos)):
        kept = df[df["v2_rejected"].isna()]
        print(f"\n  [{label}]  原始全部 vs v2 硬过滤后保留")
        print(f"    {'设置':<22} {'n':>5} {'ret_5d 均':>10} {'ret_v4 均':>10} {'ret_v4 中位':>11} {'ret_v4 胜率':>11}")
        print(f"    {'全部 (v1 接受)':<22} {len(df):>5} {df['ret_5d'].mean():>+9.2f}% {df['ret_v4'].mean():>+9.2f}% {df['ret_v4'].median():>+10.2f}% {(df['ret_v4']>0).mean()*100:>10.1f}%")
        print(f"    {'v2 硬过滤保留':<22} {len(kept):>5} {kept['ret_5d'].mean():>+9.2f}% {kept['ret_v4'].mean():>+9.2f}% {kept['ret_v4'].median():>+10.2f}% {(kept['ret_v4']>0).mean()*100:>10.1f}%")

    print(f"\n  CSV: out/ranker_v2_eval_*.csv")


if __name__ == "__main__":
    t = time.time()
    main()
    print(f"\n总耗时 {time.time()-t:.1f}s")
