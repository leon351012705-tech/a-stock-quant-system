"""
research/param_search/grid_ssb.py — 任务1 / SSB 趋势回踩确认 V5 参数搜索（补完决策1）

SSB 跟 MACD/ma_trend/boll 不一样：
  - 是 4 个里唯一**专门设计成独立策略**的（目标胜率 ≥55%）
  - **不在共振规则里**（is_valid_resonance 只检查 boll_rv ∩ {macd/ma_trend}）
  - 15 个参数，全 grid 搜不可行（~13000 组合），改用 coordinate descent：固定其他参数为 DEFAULT，
    每次只动一个，看局部敏感性

测两套出场：
  - V5 sys (-5%硬止损 + -5%移动止盈 + 20日)  ← 系统现状口径
  - V4 fix10 (-5%硬止损 + 固定 +10% 止盈 + 20日)  ← 任务1 找出来的更好出场
  SSB 性质是"趋势回踩 + 买回升"，理论上 trailing 适用（骑趋势）但 V4 验证一下。

样本内 IS 2024-01~2025-09 + 样本外 OOS 2025-10~2026-03
"""

from __future__ import annotations

import os
import sys
import time
import sqlite3
from copy import deepcopy

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import DB_PATH
from data.limit_rules import get_limit_pct
from research.param_search import _common as C
import research.strategies.ssb_bounce as SSB

IN_START,  IN_END  = "2024-01-01", "2025-09-30"
OOS_START, OOS_END = "2025-10-01", "2026-03-31"

BASELINE = deepcopy(SSB.DEFAULT_PARAMS)

# Coordinate descent：每个 knob 给 3-4 个候选值（含 baseline）
KNOB_GRID = {
    "pullback_days":        [3, 5, 7],
    "min_red_days":         [1, 2],
    "ma_touch_upper":       [1.01, 1.03, 1.05],
    "ma_touch_lower":       [0.95, 0.97, 0.99],
    "min_today_gain":       [0.0, 0.1, 0.3],
    "max_today_gain":       [3.0, 5.0, 7.0],
    "pullback_vol_ratio":   [0.80, 0.85, 0.90, 0.95],
    "today_vol_increase":   [1.00, 1.05, 1.15, 1.30],
    "max_above_ma60_pct":   [0.20, 0.30, 0.50],
}


# ════════════════════════════════════════════════════════════
#  两套出场模拟
# ════════════════════════════════════════════════════════════

def _sim(p, i, kind):
    """kind in {'V5_sys', 'V4_fix10'}; -5% 硬止损都生效"""
    n = p["n"]; buy_idx = i + 1
    if buy_idx >= n: return None
    o, h, l, c, dates = p["open"], p["high"], p["low"], p["close"], p["dates"]
    bp = o[buy_idx]
    if bp <= 0: return None
    if kind == "V5_sys":
        max_hold, use_trail, fixed_tp = 20, True, None
    else:
        max_hold, use_trail, fixed_tp = 20, False, 0.10
    peak = bp
    last_idx = min(buy_idx + max_hold - 1, n - 1)
    sell_idx = sell_price = None; reason = "到期"
    for k in range(buy_idx, last_idx + 1):
        if h[k] > peak: peak = h[k]
        sp = bp * 0.95
        if l[k] <= sp:
            sell_idx, sell_price, reason = k, sp, "止损"; break
        if use_trail:
            tp = peak * 0.95
            if (l[k] <= tp) and (peak > bp * 1.01):
                sell_idx, sell_price, reason = k, tp, "移动止盈"; break
        if fixed_tp is not None:
            tp_price = bp * (1 + fixed_tp)
            if h[k] >= tp_price:
                sell_idx, sell_price, reason = k, tp_price, "固定止盈"; break
        if k == last_idx:
            reason = "到期" if k == buy_idx + max_hold - 1 else "数据截止"
            sell_idx, sell_price = k, c[k]; break
    if sell_idx is None: return None
    net_pct  = (sell_price - bp) / bp * 100.0
    peak_pct = (peak - bp) / bp * 100.0
    return {"symbol": p.get("_sym",""), "signal_date": str(dates[i]),
            "buy_date": str(dates[buy_idx]),
            "sell_date": str(dates[sell_idx]),
            "net_pct": round(float(net_pct),3), "peak_pct": round(float(peak_pct),3),
            "exit_reason": reason, "win": 1 if net_pct > 0 else 0,
            "hold_days": int(sell_idx - buy_idx) + 1, "sell_idx": int(sell_idx)}


# ════════════════════════════════════════════════════════════
#  跑一组参数
# ════════════════════════════════════════════════════════════

def eval_combo(raw_universe, name_map, params, start, end):
    trades_v5, trades_v4 = [], []
    for sym, df_full in raw_universe.items():
        n = len(df_full)
        if n < 60: continue
        df = df_full.copy()
        # 给 generate_signals 注入板块/ST 涨跌停（它会用 df.attrs.limit_pct）
        lp = get_limit_pct(sym, name_map.get(str(sym).zfill(6), ""), df["trade_date"].iloc[-1])
        df.attrs["limit_pct"] = lp
        df.attrs["symbol"] = sym
        try:
            sig = SSB.generate_signals(df, params)
        except Exception:
            continue
        sig_arr = np.asarray(sig.to_numpy(), dtype=int)
        dates = df["trade_date"].to_numpy()
        in_range = (dates >= start) & (dates <= end)
        cand = np.where((sig_arr == 1) & in_range)[0]
        if cand.size == 0: continue

        # 同标的持仓不重叠：把 df 转成 numpy arrays 供 _sim 用
        p = {"n": n,
             "dates": dates,
             "open":  df["open"].astype(float).to_numpy(),
             "high":  df["high"].astype(float).to_numpy(),
             "low":   df["low"].astype(float).to_numpy(),
             "close": df["close"].astype(float).to_numpy(),
             "_sym":  sym}
        next_free = 0
        for i in cand:
            if i < next_free: continue
            t5 = _sim(p, int(i), "V5_sys")
            t4 = _sim(p, int(i), "V4_fix10")
            if t5 is not None: trades_v5.append(t5)
            if t4 is not None: trades_v4.append(t4)
            # 用 v5 的 sell_idx 作为 next_free（两套出场都参考同一笔买入）
            if t5 is not None:
                next_free = t5["sell_idx"] + 1
    return trades_v5, trades_v4


def summarize(trades):
    if not trades: return {"n": 0, "win_rate": 0, "avg_ret": 0, "sharpe_t": 0, "sum_ret": 0, "med_ret": 0, "avg_hold": 0}
    df = pd.DataFrame(trades)
    n = len(df); wins = int(df["win"].sum())
    avg = float(df["net_pct"].mean()); med = float(df["net_pct"].median())
    std = float(df["net_pct"].std())
    return {"n": n, "win_rate": round(wins/n*100, 1), "avg_ret": round(avg, 3),
            "med_ret": round(med, 3), "sharpe_t": round(avg/std, 3) if std > 1e-9 else 0,
            "sum_ret": round(float(df["net_pct"].sum()), 1),
            "avg_hold": round(float(df["hold_days"].mean()), 1)}


# ════════════════════════════════════════════════════════════
#  主流程
# ════════════════════════════════════════════════════════════

def make_combos():
    """Coordinate descent：baseline + 每个 knob 各偏移值。"""
    combos = [("baseline", deepcopy(BASELINE))]
    for knob, vals in KNOB_GRID.items():
        base_val = BASELINE[knob]
        for v in vals:
            if v == base_val: continue   # 避免重复 baseline
            p = deepcopy(BASELINE)
            p[knob] = v
            label = f"{knob}={v}"
            combos.append((label, p))
    return combos


def main():
    print(f"\n{'#'*72}")
    print(f"#  SSB · Coordinate descent + 双出场对比")
    print(f"#  IS  {IN_START}~{IN_END}    OOS {OOS_START}~{OOS_END}")
    print(f"#  Baseline = SSB.DEFAULT_PARAMS")
    print(f"#  Knobs: {list(KNOB_GRID.keys())}")
    print(f"{'#'*72}")

    print(f"\n加载行情 ...", flush=True); t0 = time.time()
    # 一次性把 IS+OOS 全段读进来（含 warmup + pad）
    from datetime import datetime, timedelta
    lo = (datetime.strptime(IN_START, "%Y-%m-%d") - timedelta(days=160)).strftime("%Y-%m-%d")
    hi = (datetime.strptime(OOS_END,  "%Y-%m-%d") + timedelta(days=55 )).strftime("%Y-%m-%d")
    conn = sqlite3.connect(DB_PATH)
    df_all = pd.read_sql(
        "SELECT symbol, trade_date, open, high, low, close, volume, amount, pct_change, turnover "
        "FROM daily_bars WHERE trade_date >= ? AND trade_date <= ? ORDER BY symbol, trade_date",
        conn, params=(lo, hi))
    try:
        info = pd.read_sql("SELECT symbol, name FROM stock_info", conn)
        name_map = dict(zip(info["symbol"].astype(str).str.zfill(6), info["name"].fillna("")))
    except Exception:
        name_map = {}
    conn.close()
    raw = {}
    for sym, g in df_all.groupby("symbol", sort=False):
        raw[str(sym)] = g.sort_values("trade_date").reset_index(drop=True)
    print(f"  股票池 {len(raw)} 只  {time.time()-t0:.1f}s")
    del df_all

    combos = make_combos()
    print(f"\n总计 {len(combos)} 个组合（baseline + {len(combos)-1} 个 knob 偏移）", flush=True)

    rows = []
    for k, (label, params) in enumerate(combos, 1):
        t_c = time.time()
        for region_label, s, e in [("IS", IN_START, IN_END), ("OOS", OOS_START, OOS_END)]:
            tr5, tr4 = eval_combo(raw, name_map, params, s, e)
            m5 = summarize(tr5); m4 = summarize(tr4)
            rows.append({"combo": label, "region": region_label,
                         "exit_v5_n": m5["n"], "exit_v5_win%": m5["win_rate"],
                         "exit_v5_avg%": m5["avg_ret"], "exit_v5_sh_t": m5["sharpe_t"],
                         "exit_v5_med%": m5["med_ret"], "exit_v5_sum%": m5["sum_ret"],
                         "exit_v5_hold": m5["avg_hold"],
                         "exit_v4_n": m4["n"], "exit_v4_win%": m4["win_rate"],
                         "exit_v4_avg%": m4["avg_ret"], "exit_v4_sh_t": m4["sharpe_t"],
                         "exit_v4_med%": m4["med_ret"], "exit_v4_sum%": m4["sum_ret"],
                         "exit_v4_hold": m4["avg_hold"]})
        print(f"  [{k:2d}/{len(combos)}] {label:<30}  IS_V5 n={rows[-2]['exit_v5_n']} win={rows[-2]['exit_v5_win%']}% avg={rows[-2]['exit_v5_avg%']}%  IS_V4 n={rows[-2]['exit_v4_n']} win={rows[-2]['exit_v4_win%']}% avg={rows[-2]['exit_v4_avg%']}%   ({time.time()-t_c:.1f}s)", flush=True)

    df = pd.DataFrame(rows)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "ssb_grid.csv")
    df.to_csv(out, index=False, encoding="utf-8-sig")

    # ── 报告 ──
    for region in ("IS", "OOS"):
        sub = df[df["region"] == region].copy()
        sub_sorted = sub.sort_values("exit_v4_sh_t", ascending=False).reset_index(drop=True)
        print(f"\n{'='*116}")
        print(f"  SSB · {region}  ·  按 V4 出场 sh_t 排序  ·  n_combos={len(sub_sorted)}")
        print(f"{'='*116}")
        print(f"  {'combo':<30} | {'V5_n':>5} {'V5_win%':>8} {'V5_avg%':>8} {'V5_sh_t':>8} {'V5_sum%':>9} | "
              f"{'V4_n':>5} {'V4_win%':>8} {'V4_avg%':>8} {'V4_sh_t':>8} {'V4_sum%':>9}")
        print(f"  {'-'*114}")
        for _, r in sub_sorted.iterrows():
            print(f"  {r['combo']:<30} | "
                  f"{int(r['exit_v5_n']):>5} {r['exit_v5_win%']:>7.1f}% {r['exit_v5_avg%']:>+7.2f}% {r['exit_v5_sh_t']:>+8.3f} {r['exit_v5_sum%']:>+8.1f}% | "
                  f"{int(r['exit_v4_n']):>5} {r['exit_v4_win%']:>7.1f}% {r['exit_v4_avg%']:>+7.2f}% {r['exit_v4_sh_t']:>+8.3f} {r['exit_v4_sum%']:>+8.1f}%")

    print(f"\n  CSV: {out}")


if __name__ == "__main__":
    t = time.time()
    main()
    print(f"\n总耗时 {time.time()-t:.1f}s")
