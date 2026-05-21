"""
research/param_search/eval_ranker_v3.py — v3 严格 A/B：v1 vs v2 vs v3

只动 gain_60d 重权的 v3 跟 v1/v2 头对头。判定标准：v3 必须在 OOS top-3 或 ≥0.60 阈值上
明显胜过 v1（Δmean ≥ +1pp 或 Δwin ≥ +3pp）才可推 deploy。否则保留 v1。
"""

from __future__ import annotations
import os, sys, time
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from research.param_search import _common as C
from research.param_search import resonance_exit_compare as REC
import signals.ranker     as R1
import signals.ranker_v2  as R2
import signals.ranker_v3  as R3

IN_START,  IN_END  = "2024-01-01", "2025-09-30"
OOS_START, OOS_END = "2025-10-01", "2026-03-31"
FUTURE_DAYS = 5


def _ret_v4(p, sig_idx):
    n = p["n"]; buy_idx = sig_idx + 1
    if buy_idx >= n: return None
    bp = p["open"][buy_idx]
    if bp <= 0: return None
    target, stop = bp * 1.10, bp * 0.95
    last_idx = min(buy_idx + 20 - 1, n - 1)
    for k in range(buy_idx, last_idx + 1):
        if p["low"][k] <= stop: return float((stop - bp) / bp * 100.0)
        if p["high"][k] >= target: return float((target - bp) / bp * 100.0)
        if k == last_idx: return float((p["close"][k] - bp) / bp * 100.0)
    return None


def build_scores(start, end, label):
    print(f"\n[{label}] 加载行情 ...", flush=True); t0 = time.time()
    raw, name_map, _ = C.load_universe(start, end)
    prep = C.prepare_universe(raw, name_map)
    for sym, d in raw.items():
        if sym in prep:
            prep[sym]["_turnover_raw"] = d["turnover"].astype(float).to_numpy()
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
        lo = max(0, sig_idx - 120 + 1); hi = sig_idx + 1
        df = pd.DataFrame({
            "trade_date": p["dates"][lo:hi], "open": p["open"][lo:hi], "high": p["high"][lo:hi],
            "low": p["low"][lo:hi], "close": p["close"][lo:hi],
            "volume": p["volume"][lo:hi], "amount": p["amount"][lo:hi],
            "turnover": p.get("_turnover_raw", np.full(p["n"], np.nan))[lo:hi]
                        if "_turnover_raw" in p else np.nan,
        })
        if "_turnover_raw" not in p: df["turnover"] = np.nan
        else: df["turnover"] = p["_turnover_raw"][lo:hi]
        df = df.dropna(subset=["close"]).reset_index(drop=True)
        if len(df) < 20: continue
        hits = set(s["hit_strategies"].split(","))
        hit_dates = set()
        for j in range(max(0, sig_idx - 2), sig_idx + 1):
            if p["sig_macd"][j] or p["sig_boll_rv"][j] or p["sig_ma"][j]:
                hit_dates.add(str(p["dates"][j]))
        name = name_map.get(str(sym).zfill(6), "")

        s1, _ = R1.calc_signal_score(df, hits, hit_dates, name, s["signal_date"])
        s2, i2 = R2.calc_signal_score(df, hits, hit_dates, name, s["signal_date"])
        s3, i3 = R3.calc_signal_score(df, hits, hit_dates, name, s["signal_date"])

        rv = _ret_v4(p, sig_idx)
        if rv is None: continue
        rows.append({"symbol": sym, "signal_date": s["signal_date"],
                     "gain60d": i3.get("gain_60d", 0),
                     "score_v1": s1, "score_v2": s2, "score_v3": s3,
                     "ret_v4": rv})
    df = pd.DataFrame(rows)
    print(f"  最终样本 {len(df)} 条；耗时 {time.time()-t0:.1f}s")
    return df


def topk_per_day(df, score_col, k):
    df = df[df[score_col] > 0]
    return df.sort_values(score_col, ascending=False).groupby("signal_date").head(k)


def threshold_filter(df, score_col, threshold):
    return df[df[score_col] >= threshold]


def show(picks, label):
    if len(picks) == 0:
        print(f"    {label:<14}  n=0"); return None
    win = (picks["ret_v4"] > 0).mean() * 100
    mean = picks["ret_v4"].mean()
    cum = picks["ret_v4"].sum()
    print(f"    {label:<14}  n={len(picks):>4}  win={win:>5.1f}%  mean={mean:>+5.2f}%  cum={cum:>+8.1f}%")
    return {"n": len(picks), "win": win, "mean": mean, "cum": cum}


def headhead(df, region):
    print(f"\n{'='*72}\n  {region}  n={len(df)}  ({df['signal_date'].nunique()} 个交易日)\n{'='*72}")

    print(f"\n  ── 每日 top-K 头对头 ──")
    for k in (1, 2, 3, 5):
        print(f"    K={k}:")
        r1 = show(topk_per_day(df, "score_v1", k), f"v1 top-{k}")
        r2 = show(topk_per_day(df, "score_v2", k), f"v2 top-{k}")
        r3 = show(topk_per_day(df, "score_v3", k), f"v3 top-{k}")
        if r1 and r3:
            dm = r3["mean"] - r1["mean"]; dw = r3["win"] - r1["win"]
            marker = "🟢 v3 胜" if dm > 1.0 else ("🔴 v3 输" if dm < -1.0 else "≈ 平")
            print(f"      Δ(v3-v1)         mean={dm:>+5.2f}pp  win={dw:>+5.1f}pp  {marker}")

    print(f"\n  ── 阈值筛选头对头 ──")
    for th, lab in [(0.60, "可入"), (0.65, "可入加严"), (0.70, "强推")]:
        print(f"    ≥{th} ({lab}):")
        r1 = show(threshold_filter(df, "score_v1", th), f"v1 ≥{th}")
        r2 = show(threshold_filter(df, "score_v2", th), f"v2 ≥{th}")
        r3 = show(threshold_filter(df, "score_v3", th), f"v3 ≥{th}")
        if r1 and r3:
            dm = r3["mean"] - r1["mean"]; dw = r3["win"] - r1["win"]
            marker = "🟢 v3 胜" if dm > 1.0 else ("🔴 v3 输" if dm < -1.0 else "≈ 平")
            print(f"      Δ(v3-v1)         mean={dm:>+5.2f}pp  win={dw:>+5.1f}pp  {marker}")


def main():
    print(f"\n{'#'*72}")
    print(f"#  ranker v1 vs v2 vs v3 三路 A/B（gain60d 重权 in v3）")
    print(f"#  判定: v3 必须 OOS top-3 或 ≥0.60 上 Δmean ≥+1pp 或 Δwin ≥+3pp 才 deploy")
    print(f"{'#'*72}")

    df_is  = build_scores(IN_START,  IN_END,  "IS")
    df_oos = build_scores(OOS_START, OOS_END, "OOS")
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
    df_is.to_csv(os.path.join(out_dir,  "ranker_v3_eval_is.csv"),  index=False, encoding="utf-8-sig")
    df_oos.to_csv(os.path.join(out_dir, "ranker_v3_eval_oos.csv"), index=False, encoding="utf-8-sig")

    headhead(df_is,  "IS  2024-01~2025-09")
    headhead(df_oos, "OOS 2025-10~2026-03")

    # 终判
    print(f"\n{'='*72}\n  终判：OOS 主要决策规则上 v3 是否击败 v1？\n{'='*72}")
    for k in (1, 3, 5):
        v1p = topk_per_day(df_oos, "score_v1", k); v3p = topk_per_day(df_oos, "score_v3", k)
        dm = v3p["ret_v4"].mean() - v1p["ret_v4"].mean()
        dw = (v3p["ret_v4"]>0).mean()*100 - (v1p["ret_v4"]>0).mean()*100
        print(f"  OOS top-{k}: Δmean={dm:+.2f}pp  Δwin={dw:+.1f}pp")
    for th in (0.60, 0.65):
        v1p = threshold_filter(df_oos, "score_v1", th); v3p = threshold_filter(df_oos, "score_v3", th)
        if len(v1p) == 0 or len(v3p) == 0: continue
        dm = v3p["ret_v4"].mean() - v1p["ret_v4"].mean()
        dw = (v3p["ret_v4"]>0).mean()*100 - (v1p["ret_v4"]>0).mean()*100
        print(f"  OOS ≥{th}:   Δmean={dm:+.2f}pp  Δwin={dw:+.1f}pp")


if __name__ == "__main__":
    main()
