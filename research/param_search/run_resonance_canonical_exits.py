"""
research/param_search/run_resonance_canonical_exits.py — 任务1 / B 步 close-the-loop

用 run_resonance_backtest.py 的 canonical 扫描流程拿到 cap=100 的标准共振信号集，
对每条信号同时跑 4 套出场，给出 V4 / V5 / V6 的最终对比：

  V4_fix10   = -5%硬止损 + 固定 +10% 止盈 + 20日   ← 你原版注释里 +295% 那套，grid_exits IS+OOS 都第 1
  V5_trail5  = -5%硬止损 + -5%移动止盈 + 20日       ← 你现在跑的系统现状
  V6_mid20_5 = -5%硬止损 + 收盘≥MA20 即止盈 + 5日   ← grid_exits OOS 第 3，跟 V4 几乎平
  V6_mid20_8 = -5%硬止损 + 收盘≥MA20 即止盈 + 8日   ← 之前定义的 V6（mid8）

为了避免再花 18 分钟扫，第一次跑会把 canonical 信号列表存到 out/canonical_signals_*.csv，
后续 rerun 自动加载（除非加 --rescan）。
"""

from __future__ import annotations

import os
import sys
import sqlite3
import time
from collections import defaultdict

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJ_ROOT)

from config import DB_PATH
import run_resonance_backtest as RB
from research.param_search import _common as C

# 跑哪段（沿用原版 OOS）
START_DATE, END_DATE = "2025-10-01", "2026-03-31"
TARGET = 100

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
SIG_CACHE = os.path.join(OUT_DIR, f"canonical_signals_{START_DATE}_{END_DATE}_n{TARGET}.csv")


# ════════════════════════════════════════════════════════════
#  四套出场（同接口：拿 conn + symbol + signal_date）
# ════════════════════════════════════════════════════════════

def _fetch_window(conn, symbol: str, signal_date: str,
                  pre_bars: int = 25, post_bars: int = 25) -> pd.DataFrame:
    """统一拉 signal_date 前后的 K 线（足够算 MA20 + 跑 20 日 max_hold）。"""
    return pd.read_sql(
        """
        SELECT trade_date, open, high, low, close
        FROM daily_bars
        WHERE symbol = ? AND trade_date >= (
            SELECT MIN(trade_date) FROM (
                SELECT trade_date FROM daily_bars WHERE symbol=? AND trade_date<=?
                ORDER BY trade_date DESC LIMIT ?
            )
        ) AND trade_date <= (
            SELECT MAX(trade_date) FROM (
                SELECT trade_date FROM daily_bars WHERE symbol=? AND trade_date>?
                ORDER BY trade_date ASC LIMIT ?
            )
        )
        ORDER BY trade_date
        """,
        conn, params=(symbol, symbol, signal_date, pre_bars, symbol, signal_date, post_bars),
    )


def _sim_canonical(conn, symbol: str, signal_date: str, variant: str) -> dict | None:
    """
    variant:
      'V4_fix10'   : fixed +10% TP, 20-day max
      'V5_trail5'  : -5% trailing TP, 20-day max  (== RB.simulate_trade 等价)
      'V6_mid20_5' : close>=MA20 TP, 5-day max
      'V6_mid20_8' : close>=MA20 TP, 8-day max
    所有变体都有 -5% 硬止损（盘中 low 触发）。
    """
    fut = _fetch_window(conn, symbol, signal_date)
    if len(fut) < 2:
        return None

    sig_mask = (fut["trade_date"] <= signal_date).to_numpy()
    if not sig_mask.any():
        return None
    sig_idx = int(np.where(sig_mask)[0][-1])
    buy_idx = sig_idx + 1
    if buy_idx >= len(fut):
        return None

    o = fut["open"].astype(float).to_numpy()
    h = fut["high"].astype(float).to_numpy()
    l = fut["low"].astype(float).to_numpy()
    c = fut["close"].astype(float).to_numpy()
    dates = fut["trade_date"].to_numpy()
    ma20 = pd.Series(c).rolling(20, min_periods=20).mean().to_numpy()

    buy_price = o[buy_idx]
    if buy_price <= 0:
        return None

    if variant == "V4_fix10":
        max_hold, kind = 20, "fixed"; fix = 0.10
    elif variant == "V5_trail5":
        max_hold, kind = 20, "trail"; trail = 0.05
    elif variant == "V6_mid20_5":
        max_hold, kind = 5, "target"
    elif variant == "V6_mid20_8":
        max_hold, kind = 8, "target"
    else:
        raise ValueError(variant)

    peak = buy_price
    last_idx = min(buy_idx + max_hold - 1, len(fut) - 1)
    sell_idx = sell_price = None
    reason = "到期"
    for k in range(buy_idx, last_idx + 1):
        if h[k] > peak:
            peak = h[k]
        # -5% 硬止损
        sp = buy_price * 0.95
        if l[k] <= sp:
            sell_idx, sell_price, reason = k, sp, "止损"; break
        # 止盈
        if kind == "fixed":
            tp_price = buy_price * (1.0 + fix)
            if h[k] >= tp_price:
                sell_idx, sell_price, reason = k, tp_price, "固定止盈"; break
        elif kind == "trail":
            tp_lvl = peak * (1.0 - trail)
            if (l[k] <= tp_lvl) and (peak > buy_price * 1.01):
                sell_idx, sell_price, reason = k, tp_lvl, "移动止盈"; break
        elif kind == "target":
            if (not np.isnan(ma20[k])) and (c[k] >= ma20[k]):
                sell_idx, sell_price, reason = k, c[k], "到中轨"; break
        # 到期
        if k == last_idx:
            reason = "到期" if k == buy_idx + max_hold - 1 else "数据截止"
            sell_idx, sell_price = k, c[k]; break

    if sell_idx is None:
        return None

    net_pct  = (sell_price - buy_price) / buy_price * 100.0
    peak_pct = (peak       - buy_price) / buy_price * 100.0
    return {"symbol": symbol, "signal_date": signal_date,
            "buy_date": str(dates[buy_idx]), "buy_price": round(float(buy_price), 3),
            "sell_date": str(dates[sell_idx]), "sell_price": round(float(sell_price), 3),
            "net_pct": round(float(net_pct), 3), "peak_pct": round(float(peak_pct), 3),
            "exit_reason": reason, "win": 1 if net_pct > 0 else 0,
            "hold_days": int(sell_idx - buy_idx) + 1}


# ════════════════════════════════════════════════════════════
#  扫描（与 run_resonance_v5_vs_v6.py 一致，可缓存）
# ════════════════════════════════════════════════════════════

def collect_signals_canonical(conn, all_symbols, name_map, start: str, end: str, target: int):
    if os.path.exists(SIG_CACHE):
        print(f"  ✅ 从缓存读 canonical 信号: {SIG_CACHE}", flush=True)
        df = pd.read_csv(SIG_CACHE, dtype={"symbol": str})
        return df.to_dict("records")

    print(f"  缓存不存在，开始 canonical 扫描（约 15-20 分钟）...", flush=True)
    trade_dates = RB.get_all_trade_dates(conn, start, end)
    print(f"  交易日 {len(trade_dates)} 天", flush=True)

    signals_found = []
    seen_symbols  = set()
    date_cache    = {}
    breadth_cache = {}
    skipped_weak  = 0
    scanned_dates = 0
    t0 = time.time()

    for idx, td in enumerate(trade_dates):
        ok, _ = RB.is_market_ok(conn, td, breadth_cache)
        if not ok:
            skipped_weak += 1
            continue
        scanned_dates += 1
        if td not in date_cache:
            date_cache[td] = RB.scan_one_date(conn, all_symbols, td, name_map)
        window_dates = RB.get_recent_dates_before(conn, td, RB.RESONANCE_WINDOW)
        window_hits  = [date_cache[d] for d in window_dates if d in date_cache]
        resonance    = RB.find_resonance_symbols(window_hits)

        for sym in resonance:
            if sym in seen_symbols:
                continue
            seen_symbols.add(sym)
            sym_strats = set()
            for wh in window_hits:
                for sid, sym_set in wh.items():
                    if sym in sym_set:
                        sym_strats.add(sid)
            signals_found.append({"signal_date": td, "symbol": sym,
                                  "hit_strategies": ",".join(sorted(sym_strats))})

        if (idx + 1) % 5 == 0 or idx + 1 == len(trade_dates):
            print(f"    [{idx+1:3d}/{len(trade_dates)}] {td}  累计 {len(signals_found):4d} 条  "
                  f"已扫 {scanned_dates} 天 / 跳过 {skipped_weak} 天  ({time.time()-t0:.0f}s)",
                  flush=True)

        if target and len(signals_found) >= target:
            print(f"  ✅ 已收集满 {target} 条，停止扫描", flush=True)
            break

    print(f"  共振信号 {len(signals_found)} 条；总耗时 {time.time()-t0:.1f}s", flush=True)
    pd.DataFrame(signals_found).to_csv(SIG_CACHE, index=False, encoding="utf-8-sig")
    print(f"  信号缓存到 {SIG_CACHE}（下次同区间 rerun 不用再扫）", flush=True)
    return signals_found


# ════════════════════════════════════════════════════════════
#  汇总
# ════════════════════════════════════════════════════════════

def _summarize(trades_list):
    if not trades_list:
        return C.summarize_trades(pd.DataFrame())
    df = pd.DataFrame(trades_list)
    if "hold_days" not in df.columns:
        df["hold_days"] = ((pd.to_datetime(df["sell_date"]) - pd.to_datetime(df["buy_date"]))
                           .dt.days + 1).astype(int)
    return C.summarize_trades(df)


def main():
    print(f"\n{'#'*70}")
    print(f"#  共振池 · canonical 出场对比（V4 / V5 / V6_mid5 / V6_mid8）")
    print(f"#  区间 {START_DATE} ~ {END_DATE}   target={TARGET}")
    print(f"#  扫描完全复用 run_resonance_backtest.py（cap=100 早停）")
    print(f"{'#'*70}\n")

    os.makedirs(OUT_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    info_df = pd.read_sql("SELECT symbol, name FROM stock_info ORDER BY symbol", conn)
    all_symbols = info_df["symbol"].tolist()
    name_map = dict(zip(info_df["symbol"].astype(str).str.zfill(6),
                        info_df["name"].fillna("")))
    print(f"股票池 {len(all_symbols)} 只\n")

    print(f"── 阶段 1：扫描共振信号 ──")
    sigs = collect_signals_canonical(conn, all_symbols, name_map, START_DATE, END_DATE, TARGET)
    print(f"  得 {len(sigs)} 条共振信号\n")

    print(f"── 阶段 2：4 套出场模拟同一批信号 ──")
    variants = ["V4_fix10", "V5_trail5", "V6_mid20_5", "V6_mid20_8"]
    trades_by_v: dict[str, list] = {v: [] for v in variants}
    t0 = time.time()
    for i, s in enumerate(sigs, 1):
        for v in variants:
            tr = _sim_canonical(conn, s["symbol"], s["signal_date"], v)
            if tr is not None:
                trades_by_v[v].append(tr)
        if i % 25 == 0 or i == len(sigs):
            print(f"  [{i:3d}/{len(sigs)}] 模拟中... ({time.time()-t0:.0f}s)", flush=True)
    print(f"  完成  {time.time()-t0:.1f}s\n")

    # 汇总
    rows = []
    for v in variants:
        trs = trades_by_v[v]
        m = _summarize(trs)
        er = pd.DataFrame(trs)["exit_reason"].value_counts().to_dict() if trs else {}
        rows.append({"exit": v, "n_signals": len(sigs),
                     **{k: m[k] for k in ["n","win_rate","avg_ret","med_ret","sharpe_t",
                                          "pl_ratio","avg_hold","sum_ret","port_ret",
                                          "port_maxdd","n_slot"]},
                     "er": er})

    # 打印
    print(f"\n{'='*118}")
    print(f"  CANONICAL 共振池 · 4 套出场最终对比  ·  {START_DATE}~{END_DATE}  ·  n_signals={len(sigs)}")
    print(f"{'='*118}")
    print(f"  {'exit':<13} | {'n':>4} {'win%':>6} {'avg%':>7} {'med%':>7} {'sh_t':>7} {'PL':>5} {'hold':>5} | "
          f"{'sum%':>8} {'port%':>8} {'pMDD%':>7} {'nS':>4}")
    print(f"  {'-'*114}")
    rows_sorted = sorted(rows, key=lambda r: r["sharpe_t"], reverse=True)
    for r in rows_sorted:
        pl = "" if r["pl_ratio"] is None else f"{r['pl_ratio']:.2f}"
        print(f"  {r['exit']:<13} | {r['n']:>4} {r['win_rate']:>6} {r['avg_ret']:>7} {r['med_ret']:>7} "
              f"{r['sharpe_t']:>7} {pl:>5} {r['avg_hold']:>5} | "
              f"{r['sum_ret']:>8} {r['port_ret']:>8} {r['port_maxdd']:>7} {r['n_slot']:>4}")
    print(f"\n  出场方式分布：")
    for r in rows_sorted:
        print(f"    {r['exit']:<13}: {r['er']}")

    out = os.path.join(OUT_DIR, "resonance_canonical_4exits.csv")
    pd.DataFrame([{k: v for k, v in r.items() if k != "er"} for r in rows]
                 ).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n  CSV: {out}")

    conn.close()


if __name__ == "__main__":
    t = time.time()
    main()
    print(f"\n总耗时 {time.time()-t:.1f}s")
