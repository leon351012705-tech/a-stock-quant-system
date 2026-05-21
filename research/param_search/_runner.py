"""
research/param_search/_runner.py — 通用网格搜索 + 样本内/外报告

各策略脚本只需提供 signal_fn / param_grid / baseline，剩下交给 run_grid_search：
  - 一次性把行情读进内存（样本内+外共用）
  - 样本内跑满网格 → 硬筛（_common.passes_hard_filter）→ 按 sharpe_t 排序
  - top-k + baseline 在样本外复测
  - 打印排名表 + 样本内/外对比表，写 CSV 到 out/

约定与 grid_macd.py 一致，sharpe_t 等指标见 _common.summarize_trades。
"""

from __future__ import annotations

import os
import sys
import time
import itertools

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")   # Windows GBK 兼容

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))

from research.param_search import _common as C   # noqa: E402

# ── 默认样本内/外切分（任务1 决策2 方案1）──
IN_START,  IN_END  = "2024-01-01", "2025-09-30"
OOS_START, OOS_END = "2025-10-01", "2026-03-31"

OUT_DIR = os.path.join(_HERE, "out")
os.makedirs(OUT_DIR, exist_ok=True)

# 所有评价列（_common.summarize_trades 的键）
_METRIC_COLS = ["n", "win_rate", "avg_ret", "med_ret", "sharpe_t", "pl_ratio", "avg_hold",
                "sum_ret", "port_ret", "port_maxdd", "port_sharpe", "n_slot"]


def _eval_grid(prep, signal_fn, combos, start, end, label=""):
    rows = []
    t0 = time.time()
    for k, params in enumerate(combos, 1):
        trades = C.run_param(prep, signal_fn, params, start, end)
        m = C.summarize_trades(trades)
        m.update(params)
        m["pass"] = C.passes_hard_filter(m)
        rows.append(m)
        if k % 10 == 0 or k == len(combos):
            ps = "/".join(f"{kk}{vv}" for kk, vv in params.items())
            print(f"  [{k:3d}/{len(combos)}] {time.time()-t0:6.1f}s  last:{ps} "
                  f"n={m['n']} win={m['win_rate']}% sh_t={m['sharpe_t']}", flush=True)
    return pd.DataFrame(rows)


def _fmt(v, w, nd=None):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return " " * w
    if nd is not None and isinstance(v, (int, float, np.floating)):
        return f"{v:>{w}.{nd}f}"
    return f"{str(v):>{w}}"


def _print_ranked(df, param_names, title, top=15):
    d = df.head(top)
    pcols = "".join(f"{pn:>7}" for pn in param_names)
    print(f"\n{'='*120}\n  {title}\n{'='*120}")
    print(f" {pcols} | {'n':>6} {'win%':>6} {'avg%':>7} {'med%':>7} {'sh_t':>7} {'PL':>5} {'hold':>5} | "
          f"{'sum%':>9} {'port%':>8} {'pMDD%':>7} {'pSh':>6} {'nS':>5} | pass")
    print(f" {'-'*116}")
    for _, r in d.iterrows():
        ps = "".join(_fmt(r[pn], 7) for pn in param_names)
        print(f" {ps} | {int(r['n']):>6} {_fmt(r['win_rate'],6)} {_fmt(r['avg_ret'],7)} {_fmt(r['med_ret'],7)} "
              f"{_fmt(r['sharpe_t'],7)} {_fmt(r['pl_ratio'],5)} {_fmt(r['avg_hold'],5)} | "
              f"{_fmt(r['sum_ret'],9)} {_fmt(r['port_ret'],8)} {_fmt(r['port_maxdd'],7)} {_fmt(r['port_sharpe'],6)} "
              f"{int(r['n_slot']):>5} | {'Y' if r['pass'] else '.'}")


def _match(df, params):
    m = pd.Series(True, index=df.index)
    for k, v in params.items():
        m &= (df[k] == v)
    return df[m].iloc[0]


def run_grid_search(label, signal_fn, param_grid, baseline,
                    in_start=IN_START, in_end=IN_END,
                    oos_start=OOS_START, oos_end=OOS_END,
                    top_k=3):
    """跑一个策略的完整网格搜索 + 样本外验证，打印 + 写 CSV。返回 (in_df, oos_cmp_df)。"""
    param_names = list(param_grid.keys())
    combos = [dict(zip(param_names, v)) for v in itertools.product(*param_grid.values())]
    n_combos = len(combos)

    print(f"\n{'#'*64}")
    print(f"#  {label} 参数网格搜索（任务1）")
    print(f"#  样本内 {in_start} ~ {in_end}   样本外 {oos_start} ~ {oos_end}")
    print(f"#  网格 {' × '.join(f'{pn}({len(param_grid[pn])})' for pn in param_names)} = {n_combos} 组")
    print(f"#  硬筛: 交易≥{C.HARD_MIN_TRADES} 且 胜率≥{C.HARD_MIN_WINRATE}% 且 单槽回撤≥{C.HARD_MAX_PORT_DD}%")
    print(f"#  baseline: {baseline}")
    print(f"{'#'*64}\n")

    print("加载行情到内存 ...", flush=True)
    t0 = time.time()
    raw, name_map, _ = C.load_universe(in_start, oos_end)
    prep = C.prepare_universe(raw, name_map)
    print(f"  股票池 {len(prep)} 只  耗时 {time.time()-t0:.1f}s\n", flush=True)
    del raw

    # ── 样本内 ──
    print(f"── 样本内网格搜索（{n_combos} 组）──", flush=True)
    in_df = _eval_grid(prep, signal_fn, combos, in_start, in_end, label)
    out_cols = param_names + _METRIC_COLS + ["pass"]
    in_df[out_cols].to_csv(os.path.join(OUT_DIR, f"{label}_insample.csv"), index=False, encoding="utf-8-sig")

    bl_in = _match(in_df, baseline)
    ranked    = in_df.sort_values(["sharpe_t", "avg_ret"], ascending=False).reset_index(drop=True)
    survivors = in_df[in_df["pass"]].sort_values(["sharpe_t", "avg_ret"], ascending=False).reset_index(drop=True)

    _print_ranked(ranked, param_names, f"{label} · 样本内全部 {n_combos} 组，按 sharpe_t 排序（top 15）")
    bl_ps = "/".join(f"{k}{v}" for k, v in baseline.items())
    print(f"\n  >> baseline ({bl_ps}): n={int(bl_in['n'])} win={bl_in['win_rate']}% avg={bl_in['avg_ret']}% "
          f"sh_t={bl_in['sharpe_t']} port_ret={bl_in['port_ret']}% port_mdd={bl_in['port_maxdd']}%  "
          f"pass={'Y' if bl_in['pass'] else '.'}")
    print(f"  >> 通过硬筛: {len(survivors)} / {n_combos}")

    if len(survivors) == 0:
        print(f"\n  ⚠️ {label}: 没有任何组合通过硬筛。"
              f"  → 看 out/{label}_insample.csv 的分布，再决定是否放宽硬筛 / 换出场规则 / 判定该策略单独不行。")
        return in_df, None

    # ── 样本外：baseline + top_k ──
    topk = survivors.head(top_k)
    check = [dict(baseline)] + [{pn: _coerce(r[pn]) for pn in param_names} for _, r in topk.iterrows()]
    # 去重（baseline 可能恰好是 top1）
    seen, check_u = set(), []
    for c in check:
        key = tuple(sorted(c.items()))
        if key not in seen:
            seen.add(key); check_u.append(c)
    print(f"\n── 样本外复测（baseline + top{top_k}，去重后 {len(check_u)} 组）──", flush=True)
    oos_df = _eval_grid(prep, signal_fn, check_u, oos_start, oos_end, label)

    rows = []
    labels = ["baseline"] + [f"top{i}" for i in range(1, len(topk) + 1)]
    combos_for_cmp = [dict(baseline)] + [{pn: _coerce(r[pn]) for pn in param_names} for _, r in topk.iterrows()]
    for lab, params in zip(labels, combos_for_cmp):
        ri = _match(in_df, params)
        ro = _match(oos_df, params)
        row = {"组合": lab}
        row.update({pn: params[pn] for pn in param_names})
        for tag, src in (("IS", ri), ("OOS", ro)):
            row[f"{tag}_n"]     = int(src["n"])
            row[f"{tag}_win%"]  = src["win_rate"]
            row[f"{tag}_avg%"]  = src["avg_ret"]
            row[f"{tag}_sh_t"]  = src["sharpe_t"]
            row[f"{tag}_port%"] = src["port_ret"]
            row[f"{tag}_pMDD%"] = src["port_maxdd"]
        rows.append(row)
    cmp_df = pd.DataFrame(rows)
    cmp_df.to_csv(os.path.join(OUT_DIR, f"{label}_oos.csv"), index=False, encoding="utf-8-sig")

    print(f"\n{'='*118}\n  {label} · 样本内 vs 样本外（baseline + 样本内 top{top_k}）\n{'='*118}")
    pcols = "".join(f"{pn:>7}" for pn in param_names)
    print(f"  {'组合':<10}{pcols} | {'IS:n':>6} {'win%':>6} {'avg%':>7} {'sh_t':>7} {'port%':>8} {'pMDD%':>7} | "
          f"{'OOS:n':>6} {'win%':>6} {'avg%':>7} {'sh_t':>7} {'port%':>8} {'pMDD%':>7}")
    print(f"  {'-'*114}")
    for _, r in cmp_df.iterrows():
        ps = "".join(_fmt(r[pn], 7) for pn in param_names)
        print(f"  {r['组合']:<10}{ps} | "
              f"{r['IS_n']:>6} {_fmt(r['IS_win%'],6)} {_fmt(r['IS_avg%'],7)} {_fmt(r['IS_sh_t'],7)} {_fmt(r['IS_port%'],8)} {_fmt(r['IS_pMDD%'],7)} | "
              f"{r['OOS_n']:>6} {_fmt(r['OOS_win%'],6)} {_fmt(r['OOS_avg%'],7)} {_fmt(r['OOS_sh_t'],7)} {_fmt(r['OOS_port%'],8)} {_fmt(r['OOS_pMDD%'],7)}")
    print(f"{'='*118}")
    print(f"\n  CSV: out/{label}_insample.csv   out/{label}_oos.csv")
    print(f"  解读：baseline 样本内/外是否稳定；topK 样本外是否塌方（塌方=过拟合）；"
          f"有没有候选在样本外仍稳定优于 baseline。")
    return in_df, cmp_df


def _coerce(v):
    """把从 DataFrame 取出来的 numpy 标量转回 python 原生类型，便于做参数 dict 的相等比较。"""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v
