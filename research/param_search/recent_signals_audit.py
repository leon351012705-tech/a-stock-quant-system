"""
research/param_search/recent_signals_audit.py — 最近共振信号实际表现审计

针对用户反馈"最近邮箱推送不好、开盘就掉"，把 2026-04-01 ~ 至今的全部共振信号拉出来：
  - 每条信号的 T+1 开盘价、当日 high/low/close、5 日后表现、V4 出场实际收益
  - 整体胜率、均值跟历史回测对比
  - 看是不是有规律的"开盘就掉"现象（T+1 开盘高开后阴线收盘）
  - 分组：哪类信号表现最差（按 gain60d / market 状态 / 出场原因）

直接用 in-memory 框架，秒回。
"""

from __future__ import annotations
import os, sys, sqlite3
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import DB_PATH
from research.param_search import _common as C
from research.param_search import resonance_exit_compare as REC

# 区间：最近 6 周
START, END = "2026-04-01", "2026-05-13"


def _ret_v4(p, sig_idx):
    """V4 出场（实盘建议的那套）：-5% 止损 + +10% 固定止盈 + 20 日"""
    n = p["n"]; buy_idx = sig_idx + 1
    if buy_idx >= n: return None
    bp = p["open"][buy_idx]
    if bp <= 0: return None
    target, stop = bp * 1.10, bp * 0.95
    last_idx = min(buy_idx + 20 - 1, n - 1)
    for k in range(buy_idx, last_idx + 1):
        if p["low"][k] <= stop:
            return {"net_pct": -5.0, "reason": "止损", "exit_idx": k, "hold": k - buy_idx + 1}
        if p["high"][k] >= target:
            return {"net_pct": 10.0, "reason": "固定止盈", "exit_idx": k, "hold": k - buy_idx + 1}
        if k == last_idx:
            return {"net_pct": float((p["close"][k] - bp) / bp * 100.0),
                    "reason": "到期", "exit_idx": k, "hold": k - buy_idx + 1}
    return None


def _t1_intraday(p, sig_idx):
    """T+1 当天的开盘/收盘/最高/最低 → 看开盘当天的"开盘 vs 收盘"是否阴线"""
    n = p["n"]; buy_idx = sig_idx + 1
    if buy_idx >= n: return None
    o, h, l, c = p["open"][buy_idx], p["high"][buy_idx], p["low"][buy_idx], p["close"][buy_idx]
    # T 日的收盘价（你看到信号那天）
    t_close = p["close"][sig_idx]
    # T+1 开盘相对 T 日收盘的跳空
    gap_pct = (o / t_close - 1) * 100
    # T+1 当天开盘到收盘
    open_to_close_pct = (c / o - 1) * 100
    # T+1 当天日内最低相对开盘
    open_to_low_pct = (l / o - 1) * 100
    return {"t_close": t_close, "t1_open": o, "t1_high": h, "t1_low": l, "t1_close": c,
            "gap_pct": gap_pct, "open_to_close_pct": open_to_close_pct, "open_to_low_pct": open_to_low_pct}


def main():
    print(f"\n{'#'*72}")
    print(f"#  最近共振信号实际表现审计 · {START} ~ {END}")
    print(f"{'#'*72}\n")

    print("加载行情 ...", flush=True)
    raw, name_map, _ = C.load_universe(START, END)
    prep = C.prepare_universe(raw, name_map)
    del raw

    REC._precompute_signals(prep)
    trade_dates = REC._all_trade_dates(DB_PATH, START, END)
    breadth = REC._build_breadth(prep, trade_dates)

    # 不应用 market filter，看全部共振信号（包括 market filter 拒掉的）
    breadth_all_ok = {td: {"up_ratio": 1.0, "big_drop": 0.0, "median": 1.0} for td in trade_dates}
    sigs_all = REC._scan_resonance(prep, breadth_all_ok, trade_dates, START, END)
    print(f"  全市场（不过 market filter）：{len(sigs_all)} 条共振信号")

    sigs_filtered = REC._scan_resonance(prep, breadth, trade_dates, START, END)
    print(f"  过 market filter 后：{len(sigs_filtered)} 条")

    # 对每条信号算 V4 出场 + T+1 当日表现
    rows = []
    sig_keys = {(s["symbol"], s["signal_date"]) for s in sigs_filtered}
    for s in sigs_all:
        sym = s["symbol"]; p = prep[sym]; sig_idx = s["sig_idx"]
        v4 = _ret_v4(p, sig_idx)
        t1 = _t1_intraday(p, sig_idx)
        if v4 is None or t1 is None: continue
        # 信号当日 gain60d
        if sig_idx >= 60:
            gain60d = (p["close"][sig_idx] / p["close"][sig_idx - 60] - 1) * 100
        else:
            gain60d = 0
        rows.append({
            "signal_date": s["signal_date"],
            "buy_date":    str(p["dates"][sig_idx + 1]) if sig_idx + 1 < p["n"] else "",
            "symbol": sym, "name": name_map.get(str(sym).zfill(6), "")[:10],
            "hit": s["hit_strategies"],
            "market_ok": (sym, s["signal_date"]) in sig_keys,
            "t_close": round(t1["t_close"], 2), "t1_open": round(t1["t1_open"], 2),
            "gap%":         round(t1["gap_pct"], 2),
            "t1_o2c%":      round(t1["open_to_close_pct"], 2),
            "t1_o2low%":    round(t1["open_to_low_pct"], 2),
            "gain60d":      round(gain60d, 1),
            "v4_net%":      v4["net_pct"], "v4_reason": v4["reason"], "v4_hold": v4["hold"],
        })
    df = pd.DataFrame(rows).sort_values("signal_date").reset_index(drop=True)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "recent_audit.csv")
    df.to_csv(out, index=False, encoding="utf-8-sig")

    # ── 整体表现 ──
    print(f"\n{'='*72}")
    print(f"  ① 全部共振信号 {len(df)} 条（{START} ~ {END}）")
    print(f"{'='*72}")
    print(f"  V4 出场（实盘心智）：")
    print(f"    胜率 {(df['v4_net%']>0).mean()*100:.1f}%  均收益 {df['v4_net%'].mean():+.2f}%  累计 {df['v4_net%'].sum():+.1f}%")
    print(f"    出场分布: {df['v4_reason'].value_counts().to_dict()}")

    print(f"\n  T+1 当天的[开盘就掉]现象：")
    print(f"    平均跳空(T1开盘/T收盘)：{df['gap%'].mean():+.2f}%   中位 {df['gap%'].median():+.2f}%")
    print(f"    T+1 开盘高开 (>+0.5%)：{(df['gap%']>0.5).sum()}/{len(df)} ({(df['gap%']>0.5).mean()*100:.1f}%)")
    print(f"    T+1 高开后跌（开盘高、收盘低于开盘）：{((df['gap%']>0) & (df['t1_o2c%']<0)).sum()} 条")
    print(f"    T+1 阴线（开盘到收盘下跌）：{(df['t1_o2c%']<0).sum()}/{len(df)} ({(df['t1_o2c%']<0).mean()*100:.1f}%)")
    print(f"    T+1 收盘相对开盘平均：{df['t1_o2c%'].mean():+.2f}%")
    print(f"    T+1 开盘到日内最低平均：{df['t1_o2low%'].mean():+.2f}%（负数表示开盘后跌幅）")

    # ── 与历史回测对比 ──
    print(f"\n  ② 与历史回测对比（同 V4 出场）：")
    print(f"    历史 IS 2024-01~2025-09 (n=1395)：胜率 ~50%, 均 +1.05%")
    print(f"    历史 OOS 2025-10~2026-03 (n=474)：胜率 ~51%, 均 +2.01%")
    print(f"    最近 {START}~{END}：胜率 {(df['v4_net%']>0).mean()*100:.1f}%, 均 {df['v4_net%'].mean():+.2f}%")
    if (df['v4_net%']>0).mean()*100 < 45:
        print(f"    ⚠️ 最近胜率明显低于历史 → 真实塌方")
    elif (df['v4_net%']>0).mean()*100 < 50:
        print(f"    ⚠️ 最近胜率轻微低于历史")
    else:
        print(f"    ✓ 最近胜率跟历史持平或更好")

    # ── 分组：哪类信号最差 ──
    print(f"\n{'='*72}")
    print(f"  ③ 找规律：哪类信号表现最差")
    print(f"{'='*72}")

    # 按 gain60d 分箱
    print(f"\n  按信号日 60 日涨幅分组：")
    df["gain_bin"] = pd.cut(df["gain60d"], [-99, -25, -10, 0, 10, 99], labels=["深跌(<-25%)","浅跌(-25~-10%)","小跌(-10~0%)","小涨(0~+10%)","上涨(>+10%)"])
    for b in df["gain_bin"].dropna().unique():
        sub = df[df["gain_bin"] == b]
        if len(sub) == 0: continue
        print(f"    {str(b):<18} n={len(sub):>3}  胜率{(sub['v4_net%']>0).mean()*100:>5.1f}%  均{sub['v4_net%'].mean():>+6.2f}%  T1阴{(sub['t1_o2c%']<0).mean()*100:>4.1f}%")

    # 按 T+1 跳空分组
    print(f"\n  按 T+1 开盘跳空幅度分组：")
    df["gap_bin"] = pd.cut(df["gap%"], [-99, -2, 0, 2, 5, 99], labels=["低开>2%","小幅低开","小幅高开","明显高开2-5%","暴跳>+5%"])
    for b in df["gap_bin"].dropna().unique():
        sub = df[df["gap_bin"] == b]
        if len(sub) == 0: continue
        print(f"    {str(b):<14} n={len(sub):>3}  胜率{(sub['v4_net%']>0).mean()*100:>5.1f}%  均{sub['v4_net%'].mean():>+6.2f}%")

    # 按 market filter 通过/拒掉
    print(f"\n  按 market_filter 通过 vs 拒掉：")
    for ok in [True, False]:
        sub = df[df["market_ok"] == ok]
        if len(sub) == 0: continue
        lab = "filter 通过" if ok else "filter 拒掉"
        print(f"    {lab:<15} n={len(sub):>3}  胜率{(sub['v4_net%']>0).mean()*100:>5.1f}%  均{sub['v4_net%'].mean():>+6.2f}%")

    # 最差 10 条
    print(f"\n{'='*72}")
    print(f"  ④ 最近最差的 10 条信号（按 v4_net% 升序）")
    print(f"{'='*72}")
    worst = df.nsmallest(10, "v4_net%")
    print(f"  {'信号日':<12}{'代码':<8}{'名称':<12}{'共振':<14}{'T1跳空':>8}{'T1阴线':>9}{'60日涨幅':>10}{'V4收益':>9}{'V4出场':>10}")
    for _, r in worst.iterrows():
        gap_str = f"{r['gap%']:+.2f}%"
        o2c_str = f"{r['t1_o2c%']:+.2f}%"
        gain_str = f"{r['gain60d']:+.1f}%"
        net_str = f"{r['v4_net%']:+.2f}%"
        print(f"  {r['signal_date']:<12}{r['symbol']:<8}{r['name']:<12}{r['hit']:<14}{gap_str:>8}{o2c_str:>9}{gain_str:>10}{net_str:>9}{r['v4_reason']:>10}")

    print(f"\n  CSV: {out}")


if __name__ == "__main__":
    main()
