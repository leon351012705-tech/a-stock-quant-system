"""
research/param_search/resonance_exit_compare.py — 任务1 / B 步：把"中轨止盈"用到共振池上做对比

要回答的问题：boll_rv 单独把出场从 -5%trailing 换成"中轨止盈+8日" 翻正了；
              共振池（boll_rv ∩ (macd 或 ma_trend) + 市场过滤）本来就不差，
              **换上同样的中轨止盈，共振池会更好吗，还是已经被交集筛掉了？**

设计：
  - 完全在内存里复刻 run_resonance_backtest.py 的共振扫描逻辑（参数全部取它的默认）
    * MACD: fast=12, slow=26, signal=9, zero_axis_filter=False
    * boll_rv (reversion): period=20, std_mult=2.0
    * ma_trend: fast=5, mid=20, slow=60, use_entry_filter=True
    * 共振窗口 3 个交易日, 共振规则 = boll_rv ∩ (macd 或 ma_trend)
    * 市场过滤: UP_RATIO_MIN=0.45 / WEAK=0.40 / BIG_DROP_MAX=0.20 / MEDIAN_PCT_MIN=-0.5
    * 流动性: 20日均成交额 ≥ 5000万；信号日涨跌停跳过
    * 同标的全期只取首个信号（dedup 跟原版一致）
  - 对每个共振信号同时跑两套出场：
    * sys  : -5%硬止损(low) + -5%移动止盈(peak回撤,peak>买入×1.01激活) + 持满20日收盘 ← 系统现状
    * mid8 : -5%硬止损(low) + 收盘≥MA20 即止盈 + 持满8日收盘                          ← 布林天然出场
  - 分别在样本内 2024-01~2025-09、样本外 2025-10~2026-03 上跑，给出对比

不动 run_resonance_backtest.py 任何代码；这只是诊断脚本。
"""

from __future__ import annotations

import os
import sys
import sqlite3
import time

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import DB_PATH                                  # noqa: E402
from research.param_search import _common as C              # noqa: E402

IN_START,  IN_END  = "2024-01-01", "2025-09-30"
OOS_START, OOS_END = "2025-10-01", "2026-03-31"

# 共振参数（= run_resonance_backtest.py 默认）
MACD_PARAMS    = {"fast": 12, "slow": 26, "signal": 9, "zero_axis": False}
BOLL_PARAMS    = {"period": 20, "std_mult": 2.0}
MA_PARAMS      = {"fast": 5, "mid": 20, "slow": 60, "use_entry_filter": True}
RESON_WIN_DAYS = 3
TREND_STRATS   = {"macd", "ma_trend"}

# 市场广度过滤（= run_resonance_backtest.py 默认）
UP_RATIO_MIN, UP_RATIO_WEAK = 0.45, 0.40
BIG_DROP_MAX, MEDIAN_PCT_MIN = 0.20, -0.5
BREADTH_MIN_SYMBOLS = 100   # 当日有效 pct 样本不足则当日跳过


# ════════════════════════════════════════════════════════════
#  向量化信号
# ════════════════════════════════════════════════════════════

def macd_buy(close: pd.Series, fast=12, slow=26, signal=9, zero_axis=False) -> np.ndarray:
    ef = close.ewm(span=fast, adjust=False).mean()
    es = close.ewm(span=slow, adjust=False).mean()
    diff = ef - es
    dea  = diff.ewm(span=signal, adjust=False).mean()
    prev_diff, prev_dea = diff.shift(1), dea.shift(1)
    golden = (prev_diff <= prev_dea) & (diff > dea)
    death  = (prev_diff >= prev_dea) & (diff < dea)
    buy = golden & (~death)
    if zero_axis:
        buy = buy & (diff < 0)
    return buy.fillna(False).to_numpy()


def boll_rv_buy(close: pd.Series, period=20, std_mult=2.0) -> np.ndarray:
    mid = close.rolling(period).mean()
    std = close.rolling(period).std()
    lower = mid - std_mult * std
    touched = close.shift(1) < lower.shift(1)
    bounced = close > lower
    return (touched & bounced).fillna(False).to_numpy()


def ma_trend_buy(close: pd.Series, fast=5, mid=20, slow=60, use_entry_filter=True) -> np.ndarray:
    """向量化等价 research/strategies/ma_trend.py 的 buy 分支。"""
    ma_f = close.rolling(fast).mean()
    ma10 = close.rolling(10).mean()
    ma_m = close.rolling(mid).mean()
    ma_s = close.rolling(slow).mean()
    bullish = (ma_f > ma10) & (ma10 > ma_m) & (ma_m > ma_s)
    price_above = close > ma_f
    cross_up = (ma_f > ma_m) & (ma_f.shift(1) <= ma_m.shift(1))
    buy = bullish & price_above
    if use_entry_filter:
        buy = buy & cross_up
    return buy.fillna(False).to_numpy()


# ════════════════════════════════════════════════════════════
#  出场模拟：两套出场
# ════════════════════════════════════════════════════════════

def _sim_exit(p, i: int, mid20: np.ndarray, exit_kind: str) -> dict | None:
    """
    exit_kind:
      'sys'  : -5%硬止损 + -5%移动止盈(peak>买入×1.01激活) + 持满20日收盘
      'mid8' : -5%硬止损 + 收盘≥MA20 即止盈 + 持满8日收盘
    """
    n = p["n"]
    buy_idx = i + 1
    if buy_idx >= n:
        return None
    o, h, l, c, dates = p["open"], p["high"], p["low"], p["close"], p["dates"]
    buy_price = o[buy_idx]
    if not (buy_price > 0):
        return None
    if exit_kind == "sys":
        max_hold, use_trail, use_mid = 20, True, False
    elif exit_kind == "mid8":
        max_hold, use_trail, use_mid = 8, False, True
    else:
        raise ValueError(exit_kind)
    peak = buy_price
    last_idx = min(buy_idx + max_hold - 1, n - 1)
    sell_idx = sell_price = None
    reason = "到期"
    for k in range(buy_idx, last_idx + 1):
        if h[k] > peak:
            peak = h[k]
        sp = buy_price * 0.95
        if l[k] <= sp:
            sell_idx, sell_price, reason = k, sp, "止损"
            break
        if use_trail:
            tp = peak * 0.95
            if (l[k] <= tp) and (peak > buy_price * 1.01):
                sell_idx, sell_price, reason = k, tp, "移动止盈"
                break
        if use_mid and (not np.isnan(mid20[k])) and (c[k] >= mid20[k]):
            sell_idx, sell_price, reason = k, c[k], "到中轨"
            break
        if k == last_idx:
            reason = "到期" if k == buy_idx + max_hold - 1 else "数据截止"
            sell_idx, sell_price = k, c[k]
            break
    if sell_idx is None:
        return None
    net_pct  = (sell_price - buy_price) / buy_price * 100.0
    peak_pct = (peak       - buy_price) / buy_price * 100.0
    return {"symbol": p.get("_sym", ""), "signal_date": str(dates[i]), "buy_date": str(dates[buy_idx]),
            "buy_price": round(float(buy_price), 3), "sell_date": str(dates[sell_idx]),
            "sell_price": round(float(sell_price), 3), "net_pct": round(float(net_pct), 3),
            "peak_pct": round(float(peak_pct), 3), "exit_reason": reason,
            "win": 1 if net_pct > 0 else 0, "hold_days": int(sell_idx - buy_idx) + 1, "sell_idx": int(sell_idx)}


# ════════════════════════════════════════════════════════════
#  共振扫描（一次性，两套出场共用）
# ════════════════════════════════════════════════════════════

def _all_trade_dates(db_path: str, lo: str, hi: str) -> list[str]:
    conn = sqlite3.connect(db_path)
    try:
        d = pd.read_sql(
            "SELECT DISTINCT trade_date FROM daily_bars WHERE trade_date >= ? AND trade_date <= ? ORDER BY trade_date",
            conn, params=(lo, hi),
        )
    finally:
        conn.close()
    return d["trade_date"].tolist()


def _build_breadth(prep: dict, dates: list[str]) -> dict:
    """对每个交易日计算市场广度（用 prep 里所有股票的 pct）。"""
    by_date = {td: [] for td in dates}
    for p in prep.values():
        sym_dates = p["dates"]; sym_pct = p["pct"]
        # 对齐：找出 sym_dates 中落在 dates 集合里的位置
        # 简单做法：用 dict 加速
        # 实际上 pct 数组是用 fillna(0) 处理过的，但 NaN 已不存在；为了和原版一致，
        # 严格说应只统计 pct 非缺失的；这里因 fillna(0) 已经把缺失填为 0，会轻微稀释 up_ratio。
        # 偏差小，可接受；要更严可改 prepare_universe 保留原始 NaN。
        for k, td in enumerate(sym_dates):
            lst = by_date.get(td)
            if lst is not None:
                lst.append(sym_pct[k])
    breadth = {}
    for td, arr in by_date.items():
        if len(arr) < BREADTH_MIN_SYMBOLS:
            breadth[td] = None
            continue
        a = np.asarray(arr)
        up_ratio = float((a > 0).mean())
        big_drop = float((a < -4).mean())
        median   = float(np.median(a))
        breadth[td] = {"up_ratio": up_ratio, "big_drop": big_drop, "median": median}
    return breadth


def _market_ok(b: dict) -> bool:
    if b is None:
        return False
    return ((b["up_ratio"] >= UP_RATIO_MIN and b["big_drop"] <= BIG_DROP_MAX)
            or (b["up_ratio"] >= UP_RATIO_WEAK and b["median"] >= MEDIAN_PCT_MIN))


def _precompute_signals(prep: dict):
    """为每只股票算出 macd / boll_rv / ma_trend 三个信号布尔数组，以及 MA20（出场用）。"""
    for sym, p in prep.items():
        close = pd.Series(p["close"])
        p["sig_macd"]    = macd_buy(close, **MACD_PARAMS)
        p["sig_boll_rv"] = boll_rv_buy(close, **BOLL_PARAMS)
        p["sig_ma"]      = ma_trend_buy(close, **MA_PARAMS)
        p["mid20"]       = close.rolling(20).mean().to_numpy()


def _scan_resonance(prep: dict, breadth: dict,
                    trade_dates: list[str], start: str, end: str) -> list[dict]:
    """
    扫描所有交易日，找共振信号。返回 signals 列表 [{symbol, signal_date, sig_idx, hits}]。
    全期 dedup 同标的（与 run_resonance_backtest.py 一致）。
    """
    # 把交易日映射到顺序号，方便取 3 日窗口
    date_to_pos = {td: i for i, td in enumerate(trade_dates)}

    # 为每只股票，先求出"在 trade_dates 列表里的位置序列"以及它的本地 idx
    # 然后遍历每只股票一次，把它在区间内每个日期的"命中集合"和资格(amt_ma20/limit/min_data)算出来
    # 存入 hits_by_date_pos: list[len(trade_dates)] -> {strategy_id: set(symbols)}
    n_dates = len(trade_dates)
    hits = {"macd": [set() for _ in range(n_dates)],
            "boll_rv": [set() for _ in range(n_dates)],
            "ma_trend": [set() for _ in range(n_dates)]}

    for sym, p in prep.items():
        sym_dates = p["dates"]
        # 当符号的某些日子不在全市场 trade_dates 里时跳过；找出在 trade_dates 里的 local 索引
        amt_ma20 = p["amt_ma20"]; close = p["close"]; pct = p["pct"]; lp = p["limit_pct"]
        n_sym = p["n"]
        # 资格掩码（与 _common.run_param 一致：min_data / 流动性 / 价格 / 非涨跌停）
        elig = np.zeros(n_sym, dtype=bool)
        if n_sym > C.MIN_DATA_DAYS:
            elig[C.MIN_DATA_DAYS:] = True
        elig &= (~np.isnan(amt_ma20)) & (amt_ma20 >= C.MIN_AMOUNT_W * 1e4)
        elig &= (close >= C.MIN_PRICE)
        elig &= (np.abs(pct) < (lp - 0.1))
        sm = p["sig_macd"]; sb = p["sig_boll_rv"]; sa = p["sig_ma"]
        for k in range(n_sym):
            if not elig[k]:
                continue
            td = sym_dates[k]
            pos = date_to_pos.get(td)
            if pos is None:
                continue
            if sm[k]: hits["macd"][pos].add(sym)
            if sb[k]: hits["boll_rv"][pos].add(sym)
            if sa[k]: hits["ma_trend"][pos].add(sym)

    # 扫描，构造共振信号
    signals = []
    seen = set()
    start_pos = next((i for i, td in enumerate(trade_dates) if td >= start), None)
    if start_pos is None:
        return signals
    for pos in range(start_pos, n_dates):
        td = trade_dates[pos]
        if td > end:
            break
        if not _market_ok(breadth.get(td)):
            continue
        lo_pos = max(0, pos - RESON_WIN_DAYS + 1)
        # 3 日窗口内各策略的并集（按 symbol 汇总策略集合）
        sym_strats: dict[str, set] = {}
        for w in range(lo_pos, pos + 1):
            for sid in ("macd", "boll_rv", "ma_trend"):
                for s in hits[sid][w]:
                    sym_strats.setdefault(s, set()).add(sid)
        for sym, strats in sym_strats.items():
            if "boll_rv" in strats and (strats & TREND_STRATS) and sym not in seen:
                seen.add(sym)
                # 找该信号日 td 在该 sym 的 local index
                p = prep[sym]
                arr_idx = np.searchsorted(p["dates"], td)
                # td 必须正好等于该 sym 的某天（否则该 sym 当日没数据）
                if arr_idx >= p["n"] or p["dates"][arr_idx] != td:
                    # 用窗口内任一存在的日期作为 sig 日：取窗口里最近的存在日
                    found = None
                    for w in range(pos, lo_pos - 1, -1):
                        td_w = trade_dates[w]
                        ai = np.searchsorted(p["dates"], td_w)
                        if ai < p["n"] and p["dates"][ai] == td_w:
                            found = (td_w, int(ai)); break
                    if found is None:
                        continue
                    sig_td, sig_idx = found
                else:
                    sig_td, sig_idx = td, int(arr_idx)
                signals.append({"symbol": sym, "signal_date": sig_td, "sig_idx": sig_idx,
                                "hit_strategies": ",".join(sorted(strats))})
    return signals


# ════════════════════════════════════════════════════════════
#  汇总 + 报告
# ════════════════════════════════════════════════════════════

def _summarize(trades: list[dict]) -> dict:
    df = pd.DataFrame(trades) if trades else pd.DataFrame()
    return C.summarize_trades(df)


def _run_period(prep, breadth, trade_dates, start: str, end: str, label: str):
    t0 = time.time()
    sigs = _scan_resonance(prep, breadth, trade_dates, start, end)
    scan_s = time.time() - t0
    print(f"  [{label}] 扫描完毕 {scan_s:.1f}s  共振信号 {len(sigs)} 条")

    trades_sys, trades_mid = [], []
    for s in sigs:
        p = prep[s["symbol"]]; p["_sym"] = s["symbol"]
        tr_sys = _sim_exit(p, s["sig_idx"], p["mid20"], "sys")
        tr_mid = _sim_exit(p, s["sig_idx"], p["mid20"], "mid8")
        if tr_sys is not None: trades_sys.append(tr_sys)
        if tr_mid is not None: trades_mid.append(tr_mid)
    msys = _summarize(trades_sys); mmid = _summarize(trades_mid)
    # 出场方式分布
    def er(tr):
        if not tr: return {}
        return pd.DataFrame(tr)["exit_reason"].value_counts().to_dict()
    er_sys, er_mid = er(trades_sys), er(trades_mid)
    return {"label": label, "start": start, "end": end, "n_signals": len(sigs),
            "sys": msys, "mid8": mmid, "er_sys": er_sys, "er_mid": er_mid,
            "trades_sys": trades_sys, "trades_mid": trades_mid}


def main():
    print(f"\n{'#'*66}")
    print(f"#  共振池 · 出场规则对比：sys（-5%trailing+20d） vs  mid8（中轨止盈+8d）")
    print(f"#  共振 = boll_rv ∩ (macd 或 ma_trend)，3日窗口，市场广度过滤")
    print(f"#  策略参数全部用 run_resonance_backtest.py 的默认（baseline，不动）")
    print(f"{'#'*66}\n")

    print("加载行情 ...", flush=True)
    t0 = time.time()
    raw, name_map, _ = C.load_universe(IN_START, OOS_END)
    prep = C.prepare_universe(raw, name_map)
    print(f"  股票池 {len(prep)} 只  {time.time()-t0:.1f}s")
    del raw

    print("计算每只股票的 macd / boll_rv / ma_trend 信号 + MA20 ...", flush=True)
    t0 = time.time()
    _precompute_signals(prep)
    print(f"  完成 {time.time()-t0:.1f}s")

    print("拉全市场交易日 + 计算市场广度 ...", flush=True)
    t0 = time.time()
    trade_dates = _all_trade_dates(DB_PATH, IN_START, OOS_END)
    breadth = _build_breadth(prep, trade_dates)
    n_ok = sum(1 for v in breadth.values() if _market_ok(v))
    print(f"  交易日 {len(trade_dates)} 天，过市场过滤 {n_ok} 天  {time.time()-t0:.1f}s\n")

    results = []
    for (lab, s, e) in [("样本内 IS", IN_START, IN_END), ("样本外 OOS", OOS_START, OOS_END)]:
        results.append(_run_period(prep, breadth, trade_dates, s, e, lab))

    # ── 打印对比 ──
    print(f"\n{'='*116}")
    print(f"  共振池 · sys vs mid8 出场对比（boll/macd/ma_trend 参数全部用默认 baseline）")
    print(f"{'='*116}")
    print(f"  {'period':<10} {'区间':<22} {'exit':>5} | {'n':>5} {'win%':>6} {'avg%':>7} {'med%':>7} {'sh_t':>7} {'PL':>5} {'hold':>5} | "
          f"{'sum%':>9} {'port%':>9} {'pMDD%':>7} {'nS':>4}")
    print(f"  {'-'*114}")
    for r in results:
        for ek in ("sys", "mid8"):
            m = r[ek]; pl = "" if m["pl_ratio"] is None else f"{m['pl_ratio']:.2f}"
            print(f"  {r['label']:<10} {r['start']}~{r['end']}  {ek:>5} | "
                  f"{m['n']:>5} {m['win_rate']:>6} {m['avg_ret']:>7} {m['med_ret']:>7} {m['sharpe_t']:>7} "
                  f"{pl:>5} {m['avg_hold']:>5} | {m['sum_ret']:>9} {m['port_ret']:>9} {m['port_maxdd']:>7} {m['n_slot']:>4}")
        print(f"  {'-'*114}")

    # 出场方式分布
    print(f"\n  ── 出场方式分布 ──")
    for r in results:
        print(f"  {r['label']} {r['start']}~{r['end']}  n_signals={r['n_signals']}")
        for ek, er in (("sys", r["er_sys"]), ("mid8", r["er_mid"])):
            print(f"    {ek:>5}: {er}")

    # 存 CSV
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for r in results:
        for ek in ("sys", "mid8"):
            m = r[ek]
            row = {"label": r["label"], "start": r["start"], "end": r["end"], "exit": ek,
                   "n_signals": r["n_signals"]}
            row.update({k: m[k] for k in ["n", "win_rate", "avg_ret", "med_ret", "sharpe_t",
                                          "pl_ratio", "avg_hold", "sum_ret", "port_ret",
                                          "port_maxdd", "port_sharpe", "n_slot"]})
            rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "resonance_exit_compare.csv"),
                              index=False, encoding="utf-8-sig")
    print(f"\n  CSV: out/resonance_exit_compare.csv")
    print(f"\n  解读：看同一个共振池下 mid8 的 win%/avg%/sh_t 是否明显高于 sys。"
          f"\n        若是 → 系统真该换出场；若否 → 共振交集已经把 trailing 失效的过滤掉了，不用动。")


if __name__ == "__main__":
    t = time.time()
    main()
    print(f"\n总耗时 {time.time()-t:.1f}s")
