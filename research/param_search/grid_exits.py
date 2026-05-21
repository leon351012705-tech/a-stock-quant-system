"""
research/param_search/grid_exits.py — 任务1 / B 步：出场变体小网格

在共振池上扫一组出场变体，找最优。基础信号（boll/macd/ma_trend）参数全用默认 baseline 不动，
扫描复用 resonance_exit_compare 的 in-memory 框架（快）。

出场变体（共 14 个）：
  - V4_fix10  : -5%硬止损 + 固定 +10% 止盈 + 20日   ← 你原版注释里 +295% 的那套
  - V5_trail5 : -5%硬止损 + -5%移动止盈 + 20日       ← 你现在跑的（系统现状）
  - mid20_N   : -5%硬止损 + 收盘≥MA20 即止盈 + N日   ← V6 + 持仓上限调参; N ∈ {5,8,10,15}
  - mid10_N   : 同上但目标位换成 MA10（更近的目标，可能更快但单笔更小）
  - upper_N   : 同上但目标位换成布林上轨（20,2.0）—— 最远的目标，赚到上轨级别的反弹

样本内 IS 2024-01~2025-09 + 样本外 OOS 2025-10~2026-03，两段都跑。每段同一批共振信号同时
跑这 14 个出场，按 sharpe_t 排序。

⚠️ 跟之前一致的 caveat：扫描是我的 in-memory 重写，绝对数字会跟 canonical 有差距，但变体之间的
   相对排序是稳的（同一批信号同一套 scan，只换出场）。
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

IN_START,  IN_END  = "2024-01-01", "2025-09-30"
OOS_START, OOS_END = "2025-10-01", "2026-03-31"


# ════════════════════════════════════════════════════════════
#  出场变体定义
# ════════════════════════════════════════════════════════════

# (label, target_array_name_or_None, max_hold, tp_kind, fixed_tp_pct, trail_pct)
#   tp_kind: 'target' / 'fixed' / 'trail' / 'none'
EXIT_VARIANTS = [
    ("V4_fix10",  None,      20, "fixed",  0.10, None),
    ("V5_trail5", None,      20, "trail",  None, 0.05),
    ("mid20_5",   "mid20",    5, "target", None, None),
    ("mid20_8",   "mid20",    8, "target", None, None),
    ("mid20_10",  "mid20",   10, "target", None, None),
    ("mid20_15",  "mid20",   15, "target", None, None),
    ("mid10_5",   "mid10",    5, "target", None, None),
    ("mid10_8",   "mid10",    8, "target", None, None),
    ("mid10_10",  "mid10",   10, "target", None, None),
    ("mid10_15",  "mid10",   15, "target", None, None),
    ("upper_5",   "upper",    5, "target", None, None),
    ("upper_8",   "upper",    8, "target", None, None),
    ("upper_10",  "upper",   10, "target", None, None),
    ("upper_15",  "upper",   15, "target", None, None),
]


def _precompute_extra(prep: dict):
    """除了 REC 已经算的 macd/boll_rv/ma_trend/mid20，再加 mid10 和 boll 上轨。"""
    for sym, p in prep.items():
        close = pd.Series(p["close"])
        p["mid10"] = close.rolling(10).mean().to_numpy()
        # 上轨（period=20, std_mult=2.0，与 boll baseline 一致）
        mid20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        p["upper"] = (mid20 + 2.0 * std20).to_numpy()


def _sim_variant(p: dict, i: int,
                 target_arr_name: str | None, max_hold: int, tp_kind: str,
                 fixed_tp_pct: float | None, trail_pct: float | None) -> dict | None:
    """通用单笔模拟。"""
    n = p["n"]
    buy_idx = i + 1
    if buy_idx >= n:
        return None
    o, h, l, c, dates = p["open"], p["high"], p["low"], p["close"], p["dates"]
    buy_price = o[buy_idx]
    if not (buy_price > 0):
        return None

    target = p.get(target_arr_name) if target_arr_name else None
    peak = buy_price
    last_idx = min(buy_idx + max_hold - 1, n - 1)
    sell_idx = sell_price = None
    reason = "到期"

    for k in range(buy_idx, last_idx + 1):
        if h[k] > peak:
            peak = h[k]
        # -5% 硬止损（始终生效，全部变体一致）
        sp = buy_price * 0.95
        if l[k] <= sp:
            sell_idx, sell_price, reason = k, sp, "止损"; break
        # 止盈分支
        if tp_kind == "fixed":
            tp_price = buy_price * (1.0 + fixed_tp_pct)
            if h[k] >= tp_price:                                # 盘中高点触发，按目标价成交
                sell_idx, sell_price, reason = k, tp_price, "固定止盈"; break
        elif tp_kind == "trail":
            tp_lvl = peak * (1.0 - trail_pct)
            if (l[k] <= tp_lvl) and (peak > buy_price * 1.01):
                sell_idx, sell_price, reason = k, tp_lvl, "移动止盈"; break
        elif tp_kind == "target":
            tv = target[k]
            if (not np.isnan(tv)) and (c[k] >= tv):
                sell_idx, sell_price, reason = k, c[k], "到目标"; break
        # 到期
        if k == last_idx:
            reason = "到期" if k == buy_idx + max_hold - 1 else "数据截止"
            sell_idx, sell_price = k, c[k]; break

    if sell_idx is None:
        return None

    net_pct  = (sell_price - buy_price) / buy_price * 100.0
    peak_pct = (peak       - buy_price) / buy_price * 100.0
    return {"symbol": p.get("_sym", ""), "signal_date": str(dates[i]),
            "buy_date": str(dates[buy_idx]), "buy_price": round(float(buy_price), 3),
            "sell_date": str(dates[sell_idx]), "sell_price": round(float(sell_price), 3),
            "net_pct": round(float(net_pct), 3), "peak_pct": round(float(peak_pct), 3),
            "exit_reason": reason, "win": 1 if net_pct > 0 else 0,
            "hold_days": int(sell_idx - buy_idx) + 1}


# ════════════════════════════════════════════════════════════
#  主流程
# ════════════════════════════════════════════════════════════

def _run_period(prep, breadth, trade_dates, start: str, end: str, label: str) -> list[dict]:
    t0 = time.time()
    sigs = REC._scan_resonance(prep, breadth, trade_dates, start, end)
    print(f"\n[{label} {start}~{end}]  共振信号 {len(sigs)} 条  扫描 {time.time()-t0:.1f}s", flush=True)

    rows = []
    for (vlabel, tgt, hold, kind, fix, trail) in EXIT_VARIANTS:
        trades = []
        for s in sigs:
            p = prep[s["symbol"]]; p["_sym"] = s["symbol"]
            tr = _sim_variant(p, s["sig_idx"], tgt, hold, kind, fix, trail)
            if tr is not None:
                trades.append(tr)
        m = C.summarize_trades(pd.DataFrame(trades))
        er = pd.DataFrame(trades)["exit_reason"].value_counts().to_dict() if trades else {}
        rows.append({"period": label, "start": start, "end": end, "exit": vlabel,
                     "n_signals": len(sigs),
                     **{k: m[k] for k in ["n","win_rate","avg_ret","med_ret","sharpe_t",
                                          "pl_ratio","avg_hold","sum_ret","port_ret",
                                          "port_maxdd","n_slot"]},
                     "er": er})
    return rows


def _print_period(rows, label):
    rows_p = [r for r in rows if r["period"] == label]
    rows_p = sorted(rows_p, key=lambda r: r["sharpe_t"], reverse=True)
    print(f"\n{'='*116}")
    print(f"  {label}  ·  按 sharpe_t 排序  ·  n_signals={rows_p[0]['n_signals']}")
    print(f"{'='*116}")
    print(f"  {'exit':<11} | {'n':>4} {'win%':>6} {'avg%':>7} {'med%':>7} {'sh_t':>7} {'PL':>5} {'hold':>5} | "
          f"{'sum%':>8} {'port%':>8} {'pMDD%':>7}")
    print(f"  {'-'*112}")
    for r in rows_p:
        pl = "" if r["pl_ratio"] is None else f"{r['pl_ratio']:.2f}"
        print(f"  {r['exit']:<11} | {r['n']:>4} {r['win_rate']:>6} {r['avg_ret']:>7} {r['med_ret']:>7} "
              f"{r['sharpe_t']:>7} {pl:>5} {r['avg_hold']:>5} | "
              f"{r['sum_ret']:>8} {r['port_ret']:>8} {r['port_maxdd']:>7}")


def main():
    print(f"\n{'#'*70}")
    print(f"#  共振池 · 出场变体小网格 ({len(EXIT_VARIANTS)} 个变体)")
    print(f"#  IS  {IN_START}~{IN_END}    OOS {OOS_START}~{OOS_END}")
    print(f"#  共振规则、策略参数全部用默认 baseline 不动")
    print(f"{'#'*70}\n")

    print("加载行情 ...", flush=True); t0 = time.time()
    raw, name_map, _ = C.load_universe(IN_START, OOS_END)
    prep = C.prepare_universe(raw, name_map)
    print(f"  股票池 {len(prep)} 只  {time.time()-t0:.1f}s")
    del raw

    print("预算信号 + MA20 ...", flush=True); t0 = time.time()
    REC._precompute_signals(prep)
    _precompute_extra(prep)
    print(f"  完成 {time.time()-t0:.1f}s")

    print("拉交易日 + 算市场广度 ...", flush=True); t0 = time.time()
    import sqlite3
    from config import DB_PATH
    trade_dates = REC._all_trade_dates(DB_PATH, IN_START, OOS_END)
    breadth = REC._build_breadth(prep, trade_dates)
    n_ok = sum(1 for v in breadth.values() if REC._market_ok(v))
    print(f"  交易日 {len(trade_dates)} 天，过广度 {n_ok} 天  {time.time()-t0:.1f}s")

    all_rows = []
    all_rows.extend(_run_period(prep, breadth, trade_dates, IN_START, IN_END, "IS"))
    all_rows.extend(_run_period(prep, breadth, trade_dates, OOS_START, OOS_END, "OOS"))

    _print_period(all_rows, "IS")
    _print_period(all_rows, "OOS")

    # 汇总：哪个变体在 IS 和 OOS 都靠前？
    print(f"\n{'='*116}")
    print(f"  样本内 vs 样本外 一致性（两段按 sharpe_t 排名平均；越小越稳健）")
    print(f"{'='*116}")
    rank_is  = sorted([r for r in all_rows if r["period"] == "IS"],  key=lambda r: r["sharpe_t"], reverse=True)
    rank_oos = sorted([r for r in all_rows if r["period"] == "OOS"], key=lambda r: r["sharpe_t"], reverse=True)
    rk_is  = {r["exit"]: i + 1 for i, r in enumerate(rank_is)}
    rk_oos = {r["exit"]: i + 1 for i, r in enumerate(rank_oos)}
    combo = sorted(rk_is.keys(), key=lambda e: (rk_is[e] + rk_oos[e]))
    print(f"  {'exit':<11} | {'IS rank':>8} {'OOS rank':>9} | "
          f"{'IS win%':>8} {'IS avg%':>8} {'IS sh_t':>8} | {'OOS win%':>9} {'OOS avg%':>9} {'OOS sh_t':>9}")
    print(f"  {'-'*112}")
    for e in combo:
        ri = next(r for r in all_rows if r["period"] == "IS" and r["exit"] == e)
        ro = next(r for r in all_rows if r["period"] == "OOS" and r["exit"] == e)
        print(f"  {e:<11} | {rk_is[e]:>8} {rk_oos[e]:>9} | "
              f"{ri['win_rate']:>8} {ri['avg_ret']:>8} {ri['sharpe_t']:>8} | "
              f"{ro['win_rate']:>9} {ro['avg_ret']:>9} {ro['sharpe_t']:>9}")

    # 出场方式分布（只打 IS 简表，方便看每个变体多大比例触发到目标 vs 止损 vs 到期）
    print(f"\n  IS 出场方式分布：")
    for r in [r for r in all_rows if r["period"] == "IS"]:
        print(f"    {r['exit']:<11}: {r['er']}")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "grid_exits.csv")
    pd.DataFrame([{k: v for k, v in r.items() if k != "er"} for r in all_rows]
                 ).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n  CSV: {out}")


if __name__ == "__main__":
    t = time.time()
    main()
    print(f"\n总耗时 {time.time()-t:.1f}s")
