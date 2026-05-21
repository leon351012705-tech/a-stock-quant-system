"""
research/param_search/grid_macd.py — 任务1 / MACD 金叉策略参数网格搜索（原型）

做法（决策 1 = C 的第一步：独立单策略评估，粗筛出 top 候选）：
  1. 样本内 [2024-01-01, 2025-09-30] 跑 90 组参数（fast×slow×signal×zero_axis）
  2. 硬筛：交易数≥30 且 胜率≥45% 且 单槽组合回撤≥-25%
  3. 存活组合按 sharpe_t（交易级夏普）排序
  4. 取 top-3 + baseline，在样本外 [2025-10-01, 2026-03-31] 复测，看是否仍占优

baseline = fast12/slow26/signal9/zero_axis=False
  （= run_resonance_backtest.py 里 MACD 的实际配置：generate_signals(df, zero_axis_filter=False)）

⚠️ 这一步是"MACD 单独拿出来当选股策略"的评估，不是共振池里的表现。
   共振池里的精选（决策 1 = C 的第二步）等这步方法跑通、你确认后再做。

输出：
  research/param_search/out/macd_insample.csv      （90 组全量结果）
  research/param_search/out/macd_oos.csv           （baseline + top3 的样本内/外对比）
  控制台：样本内 top15 + 样本外对比表
"""

from __future__ import annotations

import os
import sys
import time
import itertools

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")   # Windows 终端 GBK 兼容（已踩过的坑#5）

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))

from research.param_search import _common as C   # noqa: E402

# ════════════════════════════════════════════════════════════
#  区间 & 网格
# ════════════════════════════════════════════════════════════

IN_START,  IN_END  = "2024-01-01", "2025-09-30"   # 样本内（调参）
OOS_START, OOS_END = "2025-10-01", "2026-03-31"   # 样本外（验证）

GRID = {
    "fast":      [8, 10, 12, 14, 16],
    "slow":      [21, 26, 30],
    "signal":    [6, 9, 12],
    "zero_axis": [False, True],
}   # 5×3×3×2 = 90 组

BASELINE = {"fast": 12, "slow": 26, "signal": 9, "zero_axis": False}

OUT_DIR = os.path.join(_HERE, "out")
os.makedirs(OUT_DIR, exist_ok=True)


# ════════════════════════════════════════════════════════════
#  MACD 买入信号（向量化；逻辑等价 research/strategies/macd_cross.py）
# ════════════════════════════════════════════════════════════

def macd_buy_signal(p: dict, fast: int, slow: int, signal: int, zero_axis: bool) -> np.ndarray:
    close = pd.Series(p["close"])
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    diff = ema_fast - ema_slow
    dea  = diff.ewm(span=signal, adjust=False).mean()
    prev_diff, prev_dea = diff.shift(1), dea.shift(1)
    golden = (prev_diff <= prev_dea) & (diff > dea)
    death  = (prev_diff >= prev_dea) & (diff < dea)     # 原策略 sell 优先于 buy，死叉日不算买入
    buy = golden & (~death)
    if zero_axis:
        buy = buy & (diff < 0)
    return buy.fillna(False).to_numpy()


# ════════════════════════════════════════════════════════════
#  主流程
# ════════════════════════════════════════════════════════════

FMT_COLS = ["fast", "slow", "signal", "zero_axis",
            "n", "win_rate", "avg_ret", "med_ret", "sharpe_t", "pl_ratio", "avg_hold",
            "sum_ret", "port_ret", "port_maxdd", "port_sharpe", "n_slot", "pass"]


def _eval_grid(prep: dict, start: str, end: str, combos: list[dict]) -> pd.DataFrame:
    rows = []
    t0 = time.time()
    for k, params in enumerate(combos, 1):
        trades = C.run_param(prep, macd_buy_signal, params, start, end)
        m = C.summarize_trades(trades)
        m.update(params)
        m["pass"] = C.passes_hard_filter(m)
        rows.append(m)
        if k % 10 == 0 or k == len(combos):
            print(f"  [{k:3d}/{len(combos)}] {time.time()-t0:6.1f}s  "
                  f"last: f{params['fast']}/s{params['slow']}/sig{params['signal']}/z{int(params['zero_axis'])} "
                  f"n={m['n']} win={m['win_rate']}% sh_t={m['sharpe_t']}", flush=True)
    df = pd.DataFrame(rows)
    for c in FMT_COLS:
        if c not in df.columns:
            df[c] = np.nan
    return df[FMT_COLS]


def _print_table(df: pd.DataFrame, title: str, top: int | None = None):
    d = df.copy()
    if top:
        d = d.head(top)
    print(f"\n{'='*108}\n  {title}\n{'='*108}")
    print(f"  {'fast':>4} {'slow':>4} {'sig':>4} {'zero':>5} | "
          f"{'n':>5} {'win%':>6} {'avg%':>7} {'med%':>7} {'sh_t':>7} {'PL':>5} {'hold':>5} | "
          f"{'sum%':>8} {'port%':>8} {'pMDD%':>7} {'pSh':>6} {'nS':>4} | pass")
    print(f"  {'-'*108}")
    for _, r in d.iterrows():
        pl = "" if pd.isna(r['pl_ratio']) else f"{r['pl_ratio']:.2f}"
        print(f"  {int(r['fast']):>4} {int(r['slow']):>4} {int(r['signal']):>4} "
              f"{str(bool(r['zero_axis'])):>5} | "
              f"{int(r['n']):>5} {r['win_rate']:>6} {r['avg_ret']:>7} {r['med_ret']:>7} "
              f"{r['sharpe_t']:>7} {pl:>5} {r['avg_hold']:>5} | "
              f"{r['sum_ret']:>8} {r['port_ret']:>8} {r['port_maxdd']:>7} {r['port_sharpe']:>6} "
              f"{int(r['n_slot']):>4} | {'Y' if r['pass'] else '.'}")


def main():
    print(f"\n{'#'*60}")
    print(f"#  MACD 参数网格搜索（任务1 原型）")
    print(f"#  样本内 {IN_START} ~ {IN_END}   样本外 {OOS_START} ~ {OOS_END}")
    print(f"#  网格 {len(GRID['fast'])}×{len(GRID['slow'])}×{len(GRID['signal'])}×{len(GRID['zero_axis'])}"
          f" = {len(GRID['fast'])*len(GRID['slow'])*len(GRID['signal'])*len(GRID['zero_axis'])} 组")
    print(f"#  硬筛: 交易≥{C.HARD_MIN_TRADES} 且 胜率≥{C.HARD_MIN_WINRATE}% 且 单槽回撤≥{C.HARD_MAX_PORT_DD}%")
    print(f"{'#'*60}\n")

    combos = [dict(zip(GRID.keys(), v)) for v in itertools.product(*GRID.values())]

    # ── 一次性加载 + 预处理（样本内/外共用同一份内存数据）──
    print("加载行情到内存 ...", flush=True)
    t0 = time.time()
    raw, name_map, in_dates = C.load_universe(IN_START, OOS_END)   # 一次覆盖到样本外终点
    prep = C.prepare_universe(raw, name_map)
    print(f"  股票池 {len(prep)} 只（已过流动性预处理）  耗时 {time.time()-t0:.1f}s\n", flush=True)
    del raw

    # ── 样本内：跑满 90 组 ──
    print(f"── 样本内网格搜索（{len(combos)} 组）──", flush=True)
    in_df = _eval_grid(prep, IN_START, IN_END, combos)
    in_df.to_csv(os.path.join(OUT_DIR, "macd_insample.csv"), index=False, encoding="utf-8-sig")

    # baseline 行
    bl_mask = ((in_df["fast"] == BASELINE["fast"]) & (in_df["slow"] == BASELINE["slow"])
               & (in_df["signal"] == BASELINE["signal"]) & (in_df["zero_axis"] == BASELINE["zero_axis"]))
    bl_in = in_df[bl_mask].iloc[0]

    survivors = in_df[in_df["pass"]].sort_values(["sharpe_t", "avg_ret"], ascending=False).reset_index(drop=True)
    ranked    = in_df.sort_values(["sharpe_t", "avg_ret"], ascending=False).reset_index(drop=True)

    _print_table(ranked, f"样本内 全部 {len(in_df)} 组，按 sharpe_t 排序（top 15）", top=15)
    print(f"\n  >> baseline (f12/s26/sig9/z0): n={int(bl_in['n'])} win={bl_in['win_rate']}% "
          f"avg={bl_in['avg_ret']}% sh_t={bl_in['sharpe_t']} port_ret={bl_in['port_ret']}% "
          f"port_mdd={bl_in['port_maxdd']}%  pass={'Y' if bl_in['pass'] else '.'}")
    print(f"  >> 通过硬筛的组合数: {len(survivors)} / {len(in_df)}")

    if len(survivors) == 0:
        print("\n  ⚠️ 没有任何组合通过硬筛 —— 要么 MACD 单独当选股策略本就不行，"
              "要么硬筛阈值偏严。先把样本内表 CSV 拉出来看看分布再决定下一步。")
        return

    # ── 样本外：baseline + top3 复测 ──
    top3 = survivors.head(3)
    check_combos = [BASELINE] + [
        {"fast": int(r.fast), "slow": int(r.slow), "signal": int(r.signal), "zero_axis": bool(r.zero_axis)}
        for r in top3.itertuples()
    ]
    print(f"\n── 样本外复测（baseline + top3，共 {len(check_combos)} 组）──", flush=True)
    oos_df = _eval_grid(prep, OOS_START, OOS_END, check_combos)

    # 拼一张对比表：每组的样本内 vs 样本外
    def _row(label, params, src_in, src_oos):
        def pick(df):
            m = ((df["fast"] == params["fast"]) & (df["slow"] == params["slow"])
                 & (df["signal"] == params["signal"]) & (df["zero_axis"] == params["zero_axis"]))
            return df[m].iloc[0]
        ri, ro = pick(src_in), pick(src_oos)
        return {
            "组合": label,
            "fast": params["fast"], "slow": params["slow"], "signal": params["signal"],
            "zero_axis": params["zero_axis"],
            "IS_n": int(ri["n"]), "IS_win%": ri["win_rate"], "IS_avg%": ri["avg_ret"],
            "IS_sh_t": ri["sharpe_t"], "IS_port%": ri["port_ret"], "IS_pMDD%": ri["port_maxdd"],
            "OOS_n": int(ro["n"]), "OOS_win%": ro["win_rate"], "OOS_avg%": ro["avg_ret"],
            "OOS_sh_t": ro["sharpe_t"], "OOS_port%": ro["port_ret"], "OOS_pMDD%": ro["port_maxdd"],
        }

    cmp_rows = [_row("baseline", BASELINE, in_df, oos_df)]
    for i, r in enumerate(top3.itertuples(), 1):
        cmp_rows.append(_row(f"top{i}",
                             {"fast": int(r.fast), "slow": int(r.slow),
                              "signal": int(r.signal), "zero_axis": bool(r.zero_axis)},
                             in_df, oos_df))
    cmp_df = pd.DataFrame(cmp_rows)
    cmp_df.to_csv(os.path.join(OUT_DIR, "macd_oos.csv"), index=False, encoding="utf-8-sig")

    print(f"\n{'='*112}")
    print(f"  样本内 vs 样本外  对比（baseline + 样本内 top3）")
    print(f"{'='*112}")
    print(f"  {'组合':<9} {'参数(f/s/sig/z)':<16} | "
          f"{'IS:n':>5} {'win%':>6} {'avg%':>7} {'sh_t':>7} {'port%':>8} {'pMDD%':>7} | "
          f"{'OOS:n':>5} {'win%':>6} {'avg%':>7} {'sh_t':>7} {'port%':>8} {'pMDD%':>7}")
    print(f"  {'-'*112}")
    for _, r in cmp_df.iterrows():
        ps = f"{r['fast']}/{r['slow']}/{r['signal']}/{int(bool(r['zero_axis']))}"
        print(f"  {r['组合']:<9} {ps:<16} | "
              f"{r['IS_n']:>5} {r['IS_win%']:>6} {r['IS_avg%']:>7} {r['IS_sh_t']:>7} {r['IS_port%']:>8} {r['IS_pMDD%']:>7} | "
              f"{r['OOS_n']:>5} {r['OOS_win%']:>6} {r['OOS_avg%']:>7} {r['OOS_sh_t']:>7} {r['OOS_port%']:>8} {r['OOS_pMDD%']:>7}")
    print(f"{'='*112}")
    print(f"\n  CSV: {os.path.join(OUT_DIR, 'macd_insample.csv')}")
    print(f"       {os.path.join(OUT_DIR, 'macd_oos.csv')}")
    print(f"\n  解读提示：")
    print(f"   - 看 baseline 在样本内 vs 样本外是否稳定（基准锚点）")
    print(f"   - top1~3 若在样本外 sh_t/win% 明显塌方 → 过拟合，别用")
    print(f"   - 若没有候选在样本外稳定优于 baseline → 老实说 MACD 调参对单策略无明显收益")


if __name__ == "__main__":
    t_start = time.time()
    main()
    print(f"\n总耗时 {time.time()-t_start:.1f}s")
