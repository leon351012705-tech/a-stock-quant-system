"""
research/param_search/ranker_ab_backtest.py — ranker v1 vs v2 严格 A/B 回测

按用户实际使用方式做头对头：
  - 每个交易日的共振信号，按 v1 / v2 各自打分排序
  - 模拟两种实战决策规则：
    (a) "按 score 取 top-K"：每天取前 K 个，K ∈ {1, 2, 3, 5}
    (b) "按阈值过滤"：score ≥ 0.60 (v1/v2 都是"可入") / score ≥ 0.70 ("强推")
  - 真实出场（V4：-5%硬止损 + +10%固定止盈 + 20日）
  - 对比 v1 / v2 各决策规则下的：
    总笔数、胜率、均收益、累计、Δ收益（v2 - v1）

样本：IS 2024-01~2025-09 (1395 信号) + OOS 2025-10~2026-03 (467 信号)

数据源：out/ranker_v2_eval_{is,oos}.csv （已含 score_v1 / score_v2 / ret_v4 / v2_rejected）
不再重跑 in-memory 扫描。

判定标准：v2 想要替换 v1，**必须在 OOS 上至少一个常用决策规则（top-3, top-5, "可入"阈值）上明显优于 v1**，
         否则保留 v1。
"""

from __future__ import annotations
import os
import sys
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
_HERE = os.path.dirname(os.path.abspath(__file__))


def topk_per_day(df, score_col, k, score_must_be_positive=True):
    """每个 signal_date 取前 K 个；如果当天信号 < K，全取。被 v2 硬过滤（score=0）的也参与排序，
       但实际上 score=0 不会进 top-K 除非当天全是 0。"""
    if score_must_be_positive:
        df = df[df[score_col] > 0]
    grouped = df.sort_values(score_col, ascending=False).groupby("signal_date").head(k)
    return grouped


def threshold_filter(df, score_col, threshold):
    return df[df[score_col] >= threshold]


def stats(picks, label, name):
    n = len(picks)
    if n == 0:
        print(f"    {name:<18}  n=0")
        return None
    win = (picks["ret_v4"] > 0).mean() * 100
    mean = picks["ret_v4"].mean()
    cum = picks["ret_v4"].sum()
    med = picks["ret_v4"].median()
    print(f"    {name:<18}  n={n:>4}  win={win:>5.1f}%  mean={mean:>+5.2f}%  med={med:>+5.2f}%  cum={cum:>+8.1f}%")
    return {"n": n, "win": win, "mean": mean, "med": med, "cum": cum}


def ab(df, region_label):
    print(f"\n{'='*80}")
    print(f"  {region_label}  n_signals={len(df)} 条，{df['signal_date'].nunique()} 个交易日")
    print(f"  日均信号 ≈ {len(df) / df['signal_date'].nunique():.1f} 条")
    print(f"{'='*80}")

    # ── 全部信号 baseline（不排序不筛选）──
    print(f"\n  ── 不排序不筛选（全部信号取出场后的均值，作为参考）──")
    baseline = stats(df, region_label, "全部信号")

    # ── 每日 top-K 对比 ──
    print(f"\n  ── 每日按 score 取 top-K（v1 vs v2 头对头）──")
    for k in (1, 2, 3, 5):
        print(f"    K={k}:")
        v1_picks = topk_per_day(df, "score_v1", k)
        v2_picks = topk_per_day(df, "score_v2", k)
        r1 = stats(v1_picks, region_label, f"v1 top-{k}")
        r2 = stats(v2_picks, region_label, f"v2 top-{k}")
        if r1 and r2:
            dm = r2["mean"] - r1["mean"]
            dw = r2["win"] - r1["win"]
            print(f"      Δ (v2 - v1)        mean={dm:>+5.2f}pp   win={dw:>+5.1f}pp")

    # ── 阈值筛选对比 ──
    print(f"\n  ── 按阈值筛选（v1 / v2 各按自己分数阈值过滤）──")
    for th, label in [(0.60, "可入"), (0.65, "可入加严"), (0.70, "强推")]:
        print(f"    阈值 ≥ {th} ({label}):")
        v1_picks = threshold_filter(df, "score_v1", th)
        v2_picks = threshold_filter(df, "score_v2", th)
        r1 = stats(v1_picks, region_label, f"v1 ≥{th}")
        r2 = stats(v2_picks, region_label, f"v2 ≥{th}")
        if r1 and r2:
            dm = r2["mean"] - r1["mean"]
            dw = r2["win"] - r1["win"]
            print(f"      Δ (v2 - v1)        mean={dm:>+5.2f}pp   win={dw:>+5.1f}pp")

    # ── v2 硬过滤效果（独立验证）──
    if "v2_rejected" in df.columns:
        rej = df[df["v2_rejected"].notna()]
        kept = df[df["v2_rejected"].isna()]
        print(f"\n  ── v2 硬过滤踢掉的那批（独立验证）──")
        print(f"    踢掉 {len(rej)}/{len(df)} ({len(rej)/len(df)*100:.1f}%)")
        if len(rej) > 0:
            print(f"    被踢的实际表现: win={(rej['ret_v4']>0).mean()*100:.1f}%  mean={rej['ret_v4'].mean():+.2f}%")
        print(f"    保留的实际表现: win={(kept['ret_v4']>0).mean()*100:.1f}%  mean={kept['ret_v4'].mean():+.2f}%")


def main():
    out_dir = os.path.join(_HERE, "out")
    df_is  = pd.read_csv(os.path.join(out_dir, "ranker_v2_eval_is.csv"))
    df_oos = pd.read_csv(os.path.join(out_dir, "ranker_v2_eval_oos.csv"))

    print(f"\n{'#'*80}")
    print(f"#  ranker v1 vs v2  严格 A/B 头对头回测（模拟实际用法）")
    print(f"#  出场: V4 (-5%硬止损 + +10%固定止盈 + 20日)")
    print(f"#  数据: out/ranker_v2_eval_{{is,oos}}.csv")
    print(f"{'#'*80}")

    ab(df_is,  "IS  2024-01~2025-09")
    ab(df_oos, "OOS 2025-10~2026-03")

    # ── 最终判定 ──
    print(f"\n{'='*80}")
    print(f"  最终判定 (判定标准：OOS 上至少一个常用规则 top-3/top-5/≥0.60 明显胜出)")
    print(f"{'='*80}")
    verdict_rows = []
    for k in (1, 2, 3, 5):
        v1p = topk_per_day(df_oos, "score_v1", k); v2p = topk_per_day(df_oos, "score_v2", k)
        dm = v2p["ret_v4"].mean() - v1p["ret_v4"].mean() if len(v1p) and len(v2p) else 0
        dw = (v2p["ret_v4"]>0).mean()*100 - (v1p["ret_v4"]>0).mean()*100 if len(v1p) and len(v2p) else 0
        verdict_rows.append({"rule": f"top-{k}", "v1_n": len(v1p), "v2_n": len(v2p),
                             "v1_mean": v1p["ret_v4"].mean() if len(v1p) else 0,
                             "v2_mean": v2p["ret_v4"].mean() if len(v2p) else 0,
                             "Δmean": dm, "Δwin": dw})
    for th in (0.60, 0.65, 0.70):
        v1p = threshold_filter(df_oos, "score_v1", th); v2p = threshold_filter(df_oos, "score_v2", th)
        dm = v2p["ret_v4"].mean() - v1p["ret_v4"].mean() if len(v1p) and len(v2p) else 0
        dw = (v2p["ret_v4"]>0).mean()*100 - (v1p["ret_v4"]>0).mean()*100 if len(v1p) and len(v2p) else 0
        verdict_rows.append({"rule": f"≥{th}", "v1_n": len(v1p), "v2_n": len(v2p),
                             "v1_mean": v1p["ret_v4"].mean() if len(v1p) else 0,
                             "v2_mean": v2p["ret_v4"].mean() if len(v2p) else 0,
                             "Δmean": dm, "Δwin": dw})
    vdf = pd.DataFrame(verdict_rows)
    print(f"\n  OOS 头对头总结：")
    print(f"    {'rule':<10} {'v1_n':>5} {'v2_n':>5} {'v1_mean':>9} {'v2_mean':>9} {'Δmean':>8} {'Δwin':>8}")
    for _, r in vdf.iterrows():
        marker = "✅ v2 胜" if r["Δmean"] > 0.3 else ("v1 胜" if r["Δmean"] < -0.3 else "≈ 平")
        print(f"    {r['rule']:<10} {r['v1_n']:>5} {r['v2_n']:>5} {r['v1_mean']:>+8.2f}% {r['v2_mean']:>+8.2f}% {r['Δmean']:>+7.2f}pp {r['Δwin']:>+6.1f}pp  {marker}")
    vdf.to_csv(os.path.join(out_dir, "ranker_ab_verdict.csv"), index=False, encoding="utf-8-sig")
    print(f"\n  CSV: out/ranker_ab_verdict.csv")


if __name__ == "__main__":
    main()
