"""
research/param_search/probe_boll_exits.py — 诊断：布林带 reversion 到底是"参数的锅"还是"出场的锅"

背景：boll_rv 48 组参数全亏（avg -0.46% ~ -0.60%/笔），参数无杠杆。怀疑是"-5% 移动止盈"
      对均值回归反弹不友好（反弹颠簸，trailing 5% 频繁打掉）。

口径说明（用户确认）：-5% 硬止损是定死的；止盈是看趋势的移动止盈，不是固定 +X%。
      所以这里所有出场变体都保留 -5% 硬止损，只换"止盈方式"。

对比的出场变体（买入都是 T+1 开盘）：
  sys  : -5%硬止损(low) + -5%移动止盈(peak回撤, peak>买入×1.01才激活) + 持满20日收盘  ← 系统现状
  mid  : -5%硬止损(low) + 收盘≥中轨(MA-period) 即止盈 + 持满20日收盘                  ← 均值回归"自带"止盈
  mid8 : -5%硬止损(low) + 收盘≥中轨 即止盈 + 持满8日收盘                              ← 同上但更短持仓
  wide : -8%硬止损(low) + -8%移动止盈(peak回撤) + 持满30日收盘                        ← 看是不是 trailing 太紧

只跑少量 period/std_mult 组合（baseline + 样本内较优的几个）× 上面 4 个出场变体。

⚠️ 这是一次性诊断脚本，不进网格框架。
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from research.param_search import _common as C            # noqa: E402
from research.param_search.grid_boll import boll_rv_buy_signal  # noqa: E402

IN_START,  IN_END  = "2024-01-01", "2025-09-30"
OOS_START, OOS_END = "2025-10-01", "2026-03-31"

# 要测的布林参数（不多，只是为了排除"是不是某个参数+某个出场才行"）
BOLL_PARAMS = [
    {"period": 20, "std_mult": 2.0},   # baseline（同花顺默认）
    {"period": 30, "std_mult": 2.5},   # boll_rv 样本内最优（也只是 -0.46%/笔）
    {"period": 12, "std_mult": 1.8},   # 信号多的一档
    {"period": 25, "std_mult": 2.0},
]

EXIT_VARIANTS = ["sys", "mid", "mid8", "wide"]


def _mid_band(close: np.ndarray, period: int) -> np.ndarray:
    return pd.Series(close).rolling(period).mean().to_numpy()


def _simulate(p: dict, i: int, variant: str, mid: np.ndarray) -> dict | None:
    """对单标的、单信号点 i，按指定出场变体模拟一笔交易。"""
    n = p["n"]
    buy_idx = i + 1
    if buy_idx >= n:
        return None
    o, h, l, c, dates = p["open"], p["high"], p["low"], p["close"], p["dates"]
    buy_price = o[buy_idx]
    if not (buy_price > 0):
        return None

    if variant == "wide":
        hard_stop, trail, max_hold, use_trail, use_mid = -0.08, -0.08, 30, True, False
    elif variant == "sys":
        hard_stop, trail, max_hold, use_trail, use_mid = -0.05, -0.05, 20, True, False
    elif variant == "mid":
        hard_stop, trail, max_hold, use_trail, use_mid = -0.05, None, 20, False, True
    elif variant == "mid8":
        hard_stop, trail, max_hold, use_trail, use_mid = -0.05, None, 8, False, True
    else:
        raise ValueError(variant)

    peak = buy_price
    last_idx = min(buy_idx + max_hold - 1, n - 1)
    sell_idx = sell_price = None
    reason = "到期"
    for k in range(buy_idx, last_idx + 1):
        if h[k] > peak:
            peak = h[k]
        # 硬止损（盘中 low 触发）
        sp = buy_price * (1 + hard_stop)
        if l[k] <= sp:
            sell_idx, sell_price, reason = k, sp, "止损"
            break
        # 移动止盈
        if use_trail:
            tp = peak * (1 + trail)
            if (l[k] <= tp) and (peak > buy_price * 1.01):
                sell_idx, sell_price, reason = k, tp, "移动止盈"
                break
        # 到中轨止盈（收盘触发）
        if use_mid and (not np.isnan(mid[k])) and (c[k] >= mid[k]):
            sell_idx, sell_price, reason = k, c[k], "到中轨"
            break
        # 到期
        if k == last_idx:
            reason = "到期" if k == buy_idx + max_hold - 1 else "数据截止"
            sell_idx, sell_price = k, c[k]
            break
    if sell_idx is None:
        return None
    net_pct  = (sell_price - buy_price) / buy_price * 100.0
    peak_pct = (peak       - buy_price) / buy_price * 100.0
    return {
        "symbol": p.get("_sym", ""), "signal_date": str(dates[i]), "buy_date": str(dates[buy_idx]),
        "buy_price": round(float(buy_price), 3), "sell_date": str(dates[sell_idx]),
        "sell_price": round(float(sell_price), 3), "net_pct": round(float(net_pct), 3),
        "peak_pct": round(float(peak_pct), 3), "exit_reason": reason,
        "win": 1 if net_pct > 0 else 0, "hold_days": int(sell_idx - buy_idx) + 1, "sell_idx": int(sell_idx),
    }


def _run(prep, period, std_mult, variant, start, end) -> pd.DataFrame:
    trades = []
    for sym, p in prep.items():
        n = p["n"]
        dates = p["dates"]
        in_range = (dates >= start) & (dates <= end)
        if not in_range.any():
            continue
        sig = np.asarray(boll_rv_buy_signal(p, period, std_mult), dtype=bool)
        amt_ma20, close, pct, lp = p["amt_ma20"], p["close"], p["pct"], p["limit_pct"]
        idx_ok = np.zeros(n, dtype=bool)
        if n > C.MIN_DATA_DAYS:
            idx_ok[C.MIN_DATA_DAYS:] = True
        eligible = (sig & in_range & idx_ok
                    & (~np.isnan(amt_ma20)) & (amt_ma20 >= C.MIN_AMOUNT_W * 1e4)
                    & (close >= C.MIN_PRICE) & (np.abs(pct) < (lp - 0.1)))
        cand = np.where(eligible)[0]
        if cand.size == 0:
            continue
        mid = _mid_band(close, period) if variant in ("mid", "mid8") else None
        pp = dict(p); pp["_sym"] = sym
        next_free = 0
        for i in cand:
            if i < next_free:
                continue
            tr = _simulate(pp, int(i), variant, mid)
            if tr is None:
                continue
            trades.append(tr)
            next_free = tr["sell_idx"] + 1
    return pd.DataFrame(trades)


def main():
    print(f"\n{'#'*66}")
    print(f"#  布林带 reversion · 出场规则诊断")
    print(f"#  样本内 {IN_START}~{IN_END}   样本外 {OOS_START}~{OOS_END}")
    print(f"#  参数组合 {len(BOLL_PARAMS)} 个 × 出场变体 {len(EXIT_VARIANTS)} 个 = {len(BOLL_PARAMS)*len(EXIT_VARIANTS)} 跑（每跑做样本内）")
    print(f"#  出场: sys=-5%止损+-5%移动止盈+20日 | mid=-5%止损+到中轨+20日 | mid8=同mid但8日 | wide=-8%止损+-8%移动止盈+30日")
    print(f"{'#'*66}\n")

    print("加载行情 ...", flush=True)
    t0 = time.time()
    raw, name_map, _ = C.load_universe(IN_START, OOS_END)
    prep = C.prepare_universe(raw, name_map)
    print(f"  股票池 {len(prep)} 只  {time.time()-t0:.1f}s\n", flush=True)
    del raw

    rows = []
    for bp in BOLL_PARAMS:
        for v in EXIT_VARIANTS:
            tr = _run(prep, bp["period"], bp["std_mult"], v, IN_START, IN_END)
            m = C.summarize_trades(tr)
            # 出场方式分布
            er = tr["exit_reason"].value_counts().to_dict() if len(tr) else {}
            rows.append({
                "period": bp["period"], "std": bp["std_mult"], "exit": v,
                "n": m["n"], "win%": m["win_rate"], "avg%": m["avg_ret"], "med%": m["med_ret"],
                "sh_t": m["sharpe_t"], "PL": m["pl_ratio"], "hold": m["avg_hold"],
                "port%": m["port_ret"], "pMDD%": m["port_maxdd"], "nS": m["n_slot"],
                "止损": er.get("止损", 0), "移动止盈": er.get("移动止盈", 0),
                "到中轨": er.get("到中轨", 0), "到期": er.get("到期", 0) + er.get("数据截止", 0),
            })
            print(f"  period{bp['period']:>3}/std{bp['std_mult']}/{v:<5} "
                  f"n={m['n']:>6} win={m['win_rate']:>5}% avg={m['avg_ret']:>7}% sh_t={m['sharpe_t']:>7} port={m['port_ret']:>7}%",
                  flush=True)

    df = pd.DataFrame(rows)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "boll_exit_probe.csv")
    df.to_csv(out, index=False, encoding="utf-8-sig")

    print(f"\n{'='*118}")
    print(f"  样本内对比（{IN_START} ~ {IN_END}）—— 同一组布林参数下，换出场规则")
    print(f"{'='*118}")
    print(f"  {'period':>6} {'std':>5} {'exit':>6} | {'n':>6} {'win%':>6} {'avg%':>7} {'med%':>7} {'sh_t':>7} {'PL':>5} {'hold':>5} | "
          f"{'port%':>8} {'pMDD%':>7} {'nS':>5} | {'止损':>5} {'移止':>5} {'中轨':>5} {'到期':>5}")
    print(f"  {'-'*114}")
    for bp in BOLL_PARAMS:
        for v in EXIT_VARIANTS:
            r = df[(df["period"] == bp["period"]) & (df["std"] == bp["std_mult"]) & (df["exit"] == v)].iloc[0]
            print(f"  {int(r['period']):>6} {r['std']:>5} {r['exit']:>6} | {int(r['n']):>6} {r['win%']:>6} {r['avg%']:>7} {r['med%']:>7} "
                  f"{r['sh_t']:>7} {('' if pd.isna(r['PL']) else r['PL']):>5} {r['hold']:>5} | "
                  f"{r['port%']:>8} {r['pMDD%']:>7} {int(r['nS']):>5} | "
                  f"{int(r['止损']):>5} {int(r['移动止盈']):>5} {int(r['到中轨']):>5} {int(r['到期']):>5}")
        print(f"  {'-'*114}")
    print(f"\n  CSV: {out}")
    print(f"\n  看点：同一组 period/std 下，'mid'/'mid8' 的 avg%/win% 是否明显高于 'sys'。")
    print(f"        若是 → 布林的问题是'出场规则不匹配'而非'参数'；若否 → 布林 reversion 单独就是负 EV。")


if __name__ == "__main__":
    t = time.time()
    main()
    print(f"\n总耗时 {time.time()-t:.1f}s")
