"""
research/param_search/grid_boll_mid.py — 任务1 / 布林带 reversion，用"中轨止盈"重跑参数网格

前情：boll_rv 在系统现状出场（-5%硬止损+-5%移动止盈+20日）下 48 组全亏；诊断
（probe_boll_exits.py）发现换成"涨回中轨即止盈、-5%硬止损保留"后同一信号翻正，
样本内外都验证。所以现在用对的出场，重做一次 period×std_mult 的 grid search，找最优。

出场 = probe_boll_exits 的 'mid8' 变体：
  -5%硬止损（盘中 low 触发） + 收盘 ≥ 中轨(MA-period) 即止盈（当日收盘价）+ 持满 8 个交易日收盘
  （8 日上限那版在诊断里一致最稳；20 日版见 probe，差不太多）

硬筛（这一轮专用）：交易数 ≥ 30 且 胜率 ≥ 48%（已知好组合在 48~63%）。
  —— 不卡 single-slot 组合回撤，因为"全仓单槽"对~50%笔止损出局的形态太苛刻，不代表真实组合。
排序：sharpe_t 主、avg_ret 次。表里同时给 n / win% / avg%，trade-off 自己看。

样本内 2024-01~2025-09 调参 → top3 + baseline 在样本外 2025-10~2026-03 复测。
"""

from __future__ import annotations

import os
import sys
import time
import itertools

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from research.param_search import _common as C            # noqa: E402
from research.param_search import probe_boll_exits as PB   # noqa: E402  (复用 _run / _simulate / boll_rv_buy_signal)

IN_START,  IN_END  = "2024-01-01", "2025-09-30"
OOS_START, OOS_END = "2025-10-01", "2026-03-31"
EXIT_VARIANT = "mid8"

GRID = {
    "period":   [8, 10, 12, 15, 18, 20, 25, 30],
    "std_mult": [1.4, 1.6, 1.8, 2.0, 2.2, 2.5],
}   # 8 × 6 = 48
BASELINE = {"period": 20, "std_mult": 2.0}

HARD_MIN_TRADES  = 30
HARD_MIN_WINRATE = 48.0

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT_DIR, exist_ok=True)


def _passes(m: dict) -> bool:
    return m.get("n", 0) >= HARD_MIN_TRADES and m.get("win_rate", 0.0) >= HARD_MIN_WINRATE


def _eval(prep, combos, start, end):
    rows = []
    t0 = time.time()
    for k, (per, std) in enumerate(combos, 1):
        tr = PB._run(prep, per, std, EXIT_VARIANT, start, end)
        m = C.summarize_trades(tr)
        m["period"], m["std_mult"] = per, std
        m["pass"] = _passes(m)
        rows.append(m)
        if k % 8 == 0 or k == len(combos):
            print(f"  [{k:2d}/{len(combos)}] {time.time()-t0:6.1f}s  last: period{per}/std{std} "
                  f"n={m['n']} win={m['win_rate']}% avg={m['avg_ret']}% sh_t={m['sharpe_t']}", flush=True)
    return pd.DataFrame(rows)


COLS = ["period", "std_mult", "n", "win_rate", "avg_ret", "med_ret", "sharpe_t", "pl_ratio",
        "avg_hold", "sum_ret", "port_ret", "port_maxdd", "n_slot", "pass"]


def _print_tbl(df, title, top=None):
    d = df if top is None else df.head(top)
    print(f"\n{'='*112}\n  {title}\n{'='*112}")
    print(f"  {'period':>6} {'std':>5} | {'n':>6} {'win%':>6} {'avg%':>7} {'med%':>7} {'sh_t':>7} {'PL':>5} {'hold':>5} | "
          f"{'sum%':>9} {'port%':>9} {'pMDD%':>7} {'nS':>5} | pass")
    print(f"  {'-'*108}")
    for _, r in d.iterrows():
        pl = "" if pd.isna(r['pl_ratio']) else f"{r['pl_ratio']:.2f}"
        print(f"  {int(r['period']):>6} {r['std_mult']:>5} | {int(r['n']):>6} {r['win_rate']:>6} {r['avg_ret']:>7} {r['med_ret']:>7} "
              f"{r['sharpe_t']:>7} {pl:>5} {r['avg_hold']:>5} | {r['sum_ret']:>9} {r['port_ret']:>9} {r['port_maxdd']:>7} {int(r['n_slot']):>5} | "
              f"{'Y' if r['pass'] else '.'}")


def main():
    combos = list(itertools.product(GRID["period"], GRID["std_mult"]))
    print(f"\n{'#'*64}")
    print(f"#  布林带 reversion · 用'中轨止盈(mid8)'重跑参数网格（任务1）")
    print(f"#  样本内 {IN_START}~{IN_END}   样本外 {OOS_START}~{OOS_END}")
    print(f"#  网格 period({len(GRID['period'])}) × std_mult({len(GRID['std_mult'])}) = {len(combos)} 组")
    print(f"#  出场 mid8: -5%硬止损 + 收盘≥中轨止盈 + 持满8日 ; 硬筛 交易≥{HARD_MIN_TRADES} 且 胜率≥{HARD_MIN_WINRATE}%")
    print(f"#  baseline {BASELINE}")
    print(f"{'#'*64}\n")

    print("加载行情 ...", flush=True)
    t0 = time.time()
    raw, name_map, _ = C.load_universe(IN_START, OOS_END)
    prep = C.prepare_universe(raw, name_map)
    print(f"  股票池 {len(prep)} 只  {time.time()-t0:.1f}s\n", flush=True)
    del raw

    print(f"── 样本内（{len(combos)} 组）──", flush=True)
    in_df = _eval(prep, combos, IN_START, IN_END)
    in_df[COLS].to_csv(os.path.join(OUT_DIR, "boll_mid_insample.csv"), index=False, encoding="utf-8-sig")

    bl = in_df[(in_df["period"] == BASELINE["period"]) & (in_df["std_mult"] == BASELINE["std_mult"])].iloc[0]
    ranked = in_df.sort_values(["sharpe_t", "avg_ret"], ascending=False).reset_index(drop=True)
    survivors = in_df[in_df["pass"]].sort_values(["sharpe_t", "avg_ret"], ascending=False).reset_index(drop=True)
    _print_tbl(ranked, "样本内全部 48 组，按 sharpe_t 排序（top 18）", top=18)
    print(f"\n  >> baseline period20/std2.0(mid8出场): n={int(bl['n'])} win={bl['win_rate']}% avg={bl['avg_ret']}% "
          f"sh_t={bl['sharpe_t']} port_ret={bl['port_ret']}%  pass={'Y' if bl['pass'] else '.'}")
    print(f"  >> 通过硬筛: {len(survivors)} / {len(combos)}")

    if len(survivors) == 0:
        print("\n  ⚠️ 没有组合过硬筛——意外（诊断里 baseline 都到 48%）。看 out/boll_mid_insample.csv 分布。")
        return

    # ── 样本外 ──
    top3 = survivors.head(3)
    check = [(BASELINE["period"], BASELINE["std_mult"])] + [(int(r.period), float(r.std_mult)) for r in top3.itertuples()]
    seen, check_u = set(), []
    for c in check:
        if c not in seen:
            seen.add(c); check_u.append(c)
    print(f"\n── 样本外复测（baseline + top3，去重后 {len(check_u)} 组）──", flush=True)
    oos_df = _eval(prep, check_u, OOS_START, OOS_END)

    def pick(df, per, std):
        return df[(df["period"] == per) & (df["std_mult"] == std)].iloc[0]

    rows = []
    labels = ["baseline"] + [f"top{i}" for i in range(1, len(top3) + 1)]
    params = [(BASELINE["period"], BASELINE["std_mult"])] + [(int(r.period), float(r.std_mult)) for r in top3.itertuples()]
    for lab, (per, std) in zip(labels, params):
        ri, ro = pick(in_df, per, std), pick(oos_df, per, std)
        rows.append({"组合": lab, "period": per, "std_mult": std,
                     "IS_n": int(ri["n"]), "IS_win%": ri["win_rate"], "IS_avg%": ri["avg_ret"], "IS_sh_t": ri["sharpe_t"], "IS_port%": ri["port_ret"],
                     "OOS_n": int(ro["n"]), "OOS_win%": ro["win_rate"], "OOS_avg%": ro["avg_ret"], "OOS_sh_t": ro["sharpe_t"], "OOS_port%": ro["port_ret"]})
    cmp_df = pd.DataFrame(rows)
    cmp_df.to_csv(os.path.join(OUT_DIR, "boll_mid_oos.csv"), index=False, encoding="utf-8-sig")

    print(f"\n{'='*112}\n  样本内 vs 样本外（baseline + 样本内 top3，出场=mid8）\n{'='*112}")
    print(f"  {'组合':<10} {'period':>6} {'std':>5} | {'IS:n':>6} {'win%':>6} {'avg%':>7} {'sh_t':>7} {'port%':>9} | "
          f"{'OOS:n':>6} {'win%':>6} {'avg%':>7} {'sh_t':>7} {'port%':>9}")
    print(f"  {'-'*108}")
    for _, r in cmp_df.iterrows():
        print(f"  {r['组合']:<10} {int(r['period']):>6} {r['std_mult']:>5} | "
              f"{r['IS_n']:>6} {r['IS_win%']:>6} {r['IS_avg%']:>7} {r['IS_sh_t']:>7} {r['IS_port%']:>9} | "
              f"{r['OOS_n']:>6} {r['OOS_win%']:>6} {r['OOS_avg%']:>7} {r['OOS_sh_t']:>7} {r['OOS_port%']:>9}")
    print(f"{'='*112}")
    print(f"\n  CSV: out/boll_mid_insample.csv  out/boll_mid_oos.csv")
    print(f"  下一步候选：选定 period/std 后，可再小范围扫"
          f"持仓上限（5/8/10/15日）和'到中轨 vs 到上轨'止盈；然后把这套出场拿去共振回测验证。")


if __name__ == "__main__":
    t = time.time()
    main()
    print(f"\n总耗时 {time.time()-t:.1f}s")
