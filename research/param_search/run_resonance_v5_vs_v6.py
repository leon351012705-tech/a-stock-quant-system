"""
research/param_search/run_resonance_v5_vs_v6.py — 任务1 / B 步落地与验证

目的：用 run_resonance_backtest.py 的 canonical 扫描流程拿到完全一致的共振信号集，
      再分别用两套出场跑回测，得到 canonical 的 V5 vs V6 对比。

  V5 = sys   = -5%硬止损(low) + -5%移动止盈(peak回撤, peak>买入×1.01激活) + 持满 20 日收盘
       （= run_resonance_backtest.py 当前的 simulate_trade，直接复用）
  V6 = mid8  = -5%硬止损(low) + 收盘 ≥ MA20 即止盈 + 持满 8 日收盘
       （新增。-5% 硬止损沿用，止盈换成"涨回中轨"，最长持仓压到 8 日）

为什么这个脚本：
  - 我之前的 resonance_exit_compare.py 是 in-memory 重写的扫描，跟原版可能有微差（市场广度
    用 fillna(0)、没 cap=100 早停）；这个用 import 把原版的 scan_one_date / find_resonance_symbols /
    is_market_ok 全拿过来，scan 完全一致；只是给每条信号同时跑两套 simulate。
  - 跑完后，对照 resonance_exit_compare.py 的 OOS sys 行，能判断我的 in-memory 重写是否可信。

跑哪个区间：默认沿用原版的 START_DATE=2025-10-01 / END_DATE=2026-03-31 / TARGET_SIGNALS=100。
            想换区间就改下面 PERIODS / TARGET。注意：原版扫描是 SQL-per-symbol-per-day，21 个月
            样本内无 cap 跑大概 30-60 分钟，IS 那段不在本脚本做。
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
import run_resonance_backtest as RB        # 复用 canonical 扫描 + V5 出场
from research.param_search import _common as C

# ── 跑哪个区间 ──
PERIODS = [("OOS 2025-10~2026-03", "2025-10-01", "2026-03-31")]
TARGET  = 100   # 跟原版一致


# ════════════════════════════════════════════════════════════
#  V6 出场（mid8）—— 一次性拿到未来 K 线 + MA20 数组
# ════════════════════════════════════════════════════════════

def simulate_trade_mid8(conn, symbol: str, signal_date: str) -> dict | None:
    """
    与 RB.simulate_trade 同接口；出场换成 -5%硬止损 + 收盘≥MA20 即止盈 + 持满 8 日收盘。
    """
    # 取 MA20 用：需要从 signal_date 往前 ~25 个交易日的 close，加上未来 9 个交易日的 close
    future = pd.read_sql(
        """
        SELECT trade_date, open, high, low, close
        FROM daily_bars
        WHERE symbol = ? AND trade_date >= (
            SELECT MIN(trade_date) FROM (
                SELECT trade_date FROM daily_bars WHERE symbol = ? AND trade_date <= ?
                ORDER BY trade_date DESC LIMIT 25
            )
        ) AND trade_date <= (
            SELECT MAX(trade_date) FROM (
                SELECT trade_date FROM daily_bars WHERE symbol = ? AND trade_date > ?
                ORDER BY trade_date ASC LIMIT 12
            )
        )
        ORDER BY trade_date
        """,
        conn, params=(symbol, symbol, signal_date, symbol, signal_date),
    )
    if len(future) < 2:
        return None

    # signal_date 在 future 中的位置（最后一个 ≤ signal_date 的下标）
    sig_mask = (future["trade_date"] <= signal_date).to_numpy()
    if not sig_mask.any():
        return None
    sig_idx = int(np.where(sig_mask)[0][-1])
    buy_idx = sig_idx + 1
    if buy_idx >= len(future):
        return None
    buy_price = float(future.iloc[buy_idx]["open"])
    if buy_price <= 0:
        return None

    close = future["close"].astype(float).to_numpy()
    high  = future["high"].astype(float).to_numpy()
    low   = future["low"].astype(float).to_numpy()
    dates = future["trade_date"].to_numpy()
    ma20  = pd.Series(close).rolling(20, min_periods=20).mean().to_numpy()

    MAX_HOLD = 8
    peak = buy_price
    last_idx = min(buy_idx + MAX_HOLD - 1, len(future) - 1)
    sell_idx = sell_price = None
    reason = "到期"
    for k in range(buy_idx, last_idx + 1):
        if high[k] > peak:
            peak = high[k]
        sp = buy_price * 0.95
        if low[k] <= sp:
            sell_idx, sell_price, reason = k, sp, "止损"; break
        if (not np.isnan(ma20[k])) and (close[k] >= ma20[k]):
            sell_idx, sell_price, reason = k, close[k], "到中轨"; break
        if k == last_idx:
            reason = "到期" if k == buy_idx + MAX_HOLD - 1 else "数据截止"
            sell_idx, sell_price = k, close[k]; break
    if sell_idx is None:
        return None

    net_pct  = (sell_price - buy_price) / buy_price * 100.0
    peak_pct = (peak       - buy_price) / buy_price * 100.0
    return {
        "symbol": symbol, "signal_date": signal_date,
        "buy_date": str(dates[buy_idx]), "buy_price": round(float(buy_price), 3),
        "peak_price": round(float(peak), 3), "peak_pct": round(float(peak_pct), 2),
        "sell_date": str(dates[sell_idx]), "sell_price": round(float(sell_price), 3),
        "net_pct": round(float(net_pct), 2), "exit_reason": reason,
        "win": 1 if net_pct > 0 else 0, "hold_days": int(sell_idx - buy_idx) + 1,
    }


# ════════════════════════════════════════════════════════════
#  扫描共振信号（直接复用 RB 的扫描，保持 canonical）
# ════════════════════════════════════════════════════════════

def collect_signals(conn, all_symbols, name_map, start: str, end: str, target: int | None):
    """复刻 run_resonance_backtest.main() 的扫描相位（最小改动），返回 signals 列表。"""
    print(f"  扫描区间 {start} ~ {end}   target={target if target else '不限'}", flush=True)
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
    return signals_found


# ════════════════════════════════════════════════════════════
#  汇总
# ════════════════════════════════════════════════════════════

def _summarize(trades_list):
    if not trades_list:
        return C.summarize_trades(pd.DataFrame())
    df = pd.DataFrame(trades_list)
    # 字段对齐 C.summarize_trades 需要的列；V5 sim 没存 hold_days，用日期差近似
    if "hold_days" not in df.columns:
        df["hold_days"] = ((pd.to_datetime(df["sell_date"]) - pd.to_datetime(df["buy_date"]))
                           .dt.days + 1).astype(int)
    return C.summarize_trades(df)


def main():
    print(f"\n{'#'*66}")
    print(f"#  共振池 · canonical V5 vs V6 对比（扫描完全复用 run_resonance_backtest）")
    print(f"#  V5(sys)  = -5%硬止损 + -5%移动止盈 + 20日   ← 你现在跑的")
    print(f"#  V6(mid8) = -5%硬止损 + 收盘≥MA20 即止盈 + 8日   ← B 步发现的候选")
    print(f"{'#'*66}\n")

    conn = sqlite3.connect(DB_PATH)
    info_df = pd.read_sql("SELECT symbol, name FROM stock_info ORDER BY symbol", conn)
    all_symbols = info_df["symbol"].tolist()
    name_map = dict(zip(info_df["symbol"].astype(str).str.zfill(6),
                        info_df["name"].fillna("")))
    print(f"股票池 {len(all_symbols)} 只\n")

    all_rows = []
    for label, start, end in PERIODS:
        print(f"── {label} ──", flush=True)
        sigs = collect_signals(conn, all_symbols, name_map, start, end, TARGET)
        if not sigs:
            print(f"  ⚠️ 无信号，跳过")
            continue

        print(f"  模拟交易（V5 + V6 同信号）...", flush=True)
        t0 = time.time()
        trades_v5, trades_v6 = [], []
        skip = 0
        for s in sigs:
            r5 = RB.simulate_trade(conn, s["symbol"], s["signal_date"])
            r6 = simulate_trade_mid8(conn, s["symbol"], s["signal_date"])
            if r5 is not None: trades_v5.append(r5)
            if r6 is not None: trades_v6.append(r6)
            if r5 is None and r6 is None:
                skip += 1
        print(f"  完成（V5 {len(trades_v5)} 笔, V6 {len(trades_v6)} 笔, 跳过 {skip}）  {time.time()-t0:.1f}s")

        m5 = _summarize(trades_v5)
        m6 = _summarize(trades_v6)
        er5 = pd.DataFrame(trades_v5)["exit_reason"].value_counts().to_dict() if trades_v5 else {}
        er6 = pd.DataFrame(trades_v6)["exit_reason"].value_counts().to_dict() if trades_v6 else {}

        for ek, m, er in (("V5_sys", m5, er5), ("V6_mid8", m6, er6)):
            all_rows.append({"period": label, "start": start, "end": end, "exit": ek,
                             "n_signals": len(sigs),
                             **{k: m[k] for k in ["n","win_rate","avg_ret","med_ret","sharpe_t",
                                                  "pl_ratio","avg_hold","sum_ret","port_ret",
                                                  "port_maxdd","port_sharpe","n_slot"]},
                             "exit_breakdown": er})

    # ── 打印 ──
    print(f"\n{'='*118}")
    print(f"  CANONICAL V5 vs V6 共振池对比")
    print(f"{'='*118}")
    print(f"  {'period':<24} {'exit':>9} | {'n':>5} {'win%':>6} {'avg%':>7} {'med%':>7} {'sh_t':>7} {'PL':>5} {'hold':>5} | "
          f"{'sum%':>8} {'port%':>8} {'pMDD%':>7}")
    print(f"  {'-'*116}")
    for r in all_rows:
        pl = "" if r["pl_ratio"] is None else f"{r['pl_ratio']:.2f}"
        print(f"  {r['period']:<24} {r['exit']:>9} | {r['n']:>5} {r['win_rate']:>6} {r['avg_ret']:>7} {r['med_ret']:>7} "
              f"{r['sharpe_t']:>7} {pl:>5} {r['avg_hold']:>5} | {r['sum_ret']:>8} {r['port_ret']:>8} {r['port_maxdd']:>7}")
    print(f"\n  出场方式分布：")
    for r in all_rows:
        print(f"    {r['period']} {r['exit']}: {r['exit_breakdown']}")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "resonance_v5_vs_v6.csv")
    pd.DataFrame([{k: v for k, v in r.items() if k != "exit_breakdown"} for r in all_rows]
                 ).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n  CSV: {out}")

    conn.close()


if __name__ == "__main__":
    t = time.time()
    main()
    print(f"\n总耗时 {time.time()-t:.1f}s")
