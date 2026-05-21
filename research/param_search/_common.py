"""
research/param_search/_common.py — 参数网格搜索的共用底座

核心思路：把要回测的区间（含指标预热 + 出场所需的未来 K 线）一次性读进内存，
之后每组参数只做"算信号 + 模拟交易"，避免反复查 SQLite。

模拟交易约定（与 run_resonance_backtest.py 的 simulate_trade 完全一致，便于对比）：
  - T+1 开盘买入（A股 T+1）
  - 硬止损 STOP_LOSS_PCT = -5%（当日 low ≤ 买入价×0.95 触发，成交价记为 0.95×买入价）
  - 移动止盈 TRAIL_PCT = -5%（持仓最高价回撤 5%，且最高价 > 买入价×1.01 才激活）
  - 持满 MAX_HOLD_DAYS = 20 个交易日强制收盘出场
  - 信号日若涨跌停（板块/ST 感知）→ 跳过（T+1 难以开盘成交）
  - 同一标的持仓期间不重复开仓（出场后下一根 K 线起可再进）—— 比共振回测的"每标的只取首个信号"
    更接近真实，也能拿到更大样本

无前视（lookahead）保证：
  - 算信号的 EMA / rolling 在第 i 天只用 ≤ i 的数据
  - 信号日严格限制在评估区间内；模拟交易往后看 K 线是有意为之（要知道结果）

⚠️ 已知限制（与现有 baseline 共享，不在本任务范围内修）：
  ST 涨跌停判定用 stock_info 当前 name，历史上的 ST 状态可能不同 → 轻微前视。
"""

from __future__ import annotations

import os
import sys
import sqlite3
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# 让 `python research/param_search/xxx.py` 直接跑得起来
_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

from config import DB_PATH                                  # noqa: E402
from data.limit_rules import get_limit_pct, is_limit_move   # noqa: E402

# ── 模拟交易参数（与 run_resonance_backtest.py 对齐）──
STOP_LOSS_PCT = -0.05
TRAIL_PCT     = -0.05
MAX_HOLD_DAYS = 20

# ── 选股门槛 ──
MIN_AMOUNT_W  = 5000      # 20 日均成交额 ≥ 5000 万元（= run_resonance_backtest 的 MIN_AMOUNT）
MIN_PRICE     = 2.0       # 收盘价 ≥ 2 元
MIN_DATA_DAYS = 60        # 信号点之前至少要有的历史 K 线根数

# ── 加载窗口的两侧富余（日历天）──
WARMUP_DAYS = 160         # 评估区间起点往前多读，给最长均线（MA80 之类）预热
PAD_DAYS    = 55          # 评估区间终点往后多读，给最后一批信号留 ~20 个交易日出场


def _shift(date_str: str, days: int) -> str:
    return (datetime.strptime(date_str, "%Y-%m-%d") + timedelta(days=days)).strftime("%Y-%m-%d")


# ════════════════════════════════════════════════════════════
#  数据加载 & 预处理
# ════════════════════════════════════════════════════════════

def load_universe(start: str, end: str,
                  warmup_days: int = WARMUP_DAYS,
                  pad_days: int = PAD_DAYS,
                  db_path: str = DB_PATH) -> tuple[dict, dict, list]:
    """
    一次性把 [start-warmup, end+pad] 区间所有股票的日线读进内存。

    返回:
        raw_universe : dict[symbol] -> DataFrame（升序, reset_index）
        name_map     : dict[symbol(6位)] -> name
        sample_dates : list，落在 [start, end] 内的交易日（升序）
    """
    lo = _shift(start, -warmup_days)
    hi = _shift(end,    pad_days)
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql(
            """
            SELECT symbol, trade_date, open, high, low, close, volume, amount, pct_change, turnover
            FROM daily_bars
            WHERE trade_date >= ? AND trade_date <= ?
            ORDER BY symbol, trade_date
            """,
            conn, params=(lo, hi),
        )
        try:
            info = pd.read_sql("SELECT symbol, name FROM stock_info", conn)
            name_map = dict(zip(info["symbol"].astype(str).str.zfill(6),
                                info["name"].fillna("")))
        except Exception:
            name_map = {}
    finally:
        conn.close()

    sample_dates = sorted(
        df.loc[(df["trade_date"] >= start) & (df["trade_date"] <= end), "trade_date"].unique().tolist()
    )

    raw_universe: dict[str, pd.DataFrame] = {}
    for sym, g in df.groupby("symbol", sort=False):
        raw_universe[str(sym)] = g.sort_values("trade_date").reset_index(drop=True)
    return raw_universe, name_map, sample_dates


def prepare_universe(raw_universe: dict, name_map: dict) -> dict:
    """
    把每只股票预处理成 numpy 数组包，后续每组参数复用（流动性/涨跌停只算一次）。

    返回: dict[symbol] -> {
        n, dates(np.str), open, high, low, close, volume, amount, pct,
        amt_ma20, limit_pct, name
    }
    """
    prep: dict[str, dict] = {}
    for sym, g in raw_universe.items():
        n = len(g)
        if n < MIN_DATA_DAYS + 3:
            continue
        amt = g["amount"].astype(float)
        amt_ma20 = amt.rolling(20, min_periods=20).mean().to_numpy()
        name = name_map.get(str(sym).zfill(6), "")
        # 板块固定 → limit_pct 对全段视为常数（区间≥2024，创业板早已 20%；ST 用当前 name）
        limit_pct = get_limit_pct(sym, name, g["trade_date"].iloc[-1])
        prep[sym] = {
            "n":         n,
            "dates":     g["trade_date"].to_numpy(),
            "open":      g["open"].astype(float).to_numpy(),
            "high":      g["high"].astype(float).to_numpy(),
            "low":       g["low"].astype(float).to_numpy(),
            "close":     g["close"].astype(float).to_numpy(),
            "volume":    g["volume"].astype(float).to_numpy(),
            "amount":    amt.to_numpy(),
            "pct":       g["pct_change"].astype(float).fillna(0.0).to_numpy(),
            "amt_ma20":  amt_ma20,
            "limit_pct": float(limit_pct),
            "name":      name,
        }
    return prep


# ════════════════════════════════════════════════════════════
#  单标的模拟交易
# ════════════════════════════════════════════════════════════

def _simulate_trade(p: dict, i: int) -> dict | None:
    """
    p : prepare_universe 产出的单标的 dict
    i : 信号点行号（第 i 天收盘出信号，T+1 = 第 i+1 天开盘买入）
    返回 trade dict（含 sell_idx，用于持仓不重叠判断）或 None
    """
    n = p["n"]
    buy_idx = i + 1
    if buy_idx >= n:
        return None
    o, h, l, c = p["open"], p["high"], p["low"], p["close"]
    dates = p["dates"]
    buy_price = o[buy_idx]
    if not (buy_price > 0):
        return None

    peak = buy_price
    last_idx = min(buy_idx + MAX_HOLD_DAYS - 1, n - 1)
    sell_idx = None
    sell_price = None
    reason = "到期"

    for k in range(buy_idx, last_idx + 1):
        if h[k] > peak:
            peak = h[k]
        stop_price = buy_price * (1 + STOP_LOSS_PCT)
        if l[k] <= stop_price:
            sell_idx, sell_price, reason = k, stop_price, "止损"
            break
        trail_price = peak * (1 + TRAIL_PCT)
        if (l[k] <= trail_price) and (peak > buy_price * 1.01):
            sell_idx, sell_price, reason = k, trail_price, "移动止盈"
            break
        if k == last_idx:
            reason = "到期" if k == buy_idx + MAX_HOLD_DAYS - 1 else "数据截止"
            sell_idx, sell_price = k, c[k]
            break

    if sell_idx is None:
        return None

    net_pct  = (sell_price - buy_price) / buy_price * 100.0
    peak_pct = (peak       - buy_price) / buy_price * 100.0
    return {
        "symbol":      p.get("_sym", ""),
        "signal_date": str(dates[i]),
        "buy_date":    str(dates[buy_idx]),
        "buy_price":   round(float(buy_price), 3),
        "sell_date":   str(dates[sell_idx]),
        "sell_price":  round(float(sell_price), 3),
        "net_pct":     round(float(net_pct), 3),
        "peak_pct":    round(float(peak_pct), 3),
        "exit_reason": reason,
        "win":         1 if net_pct > 0 else 0,
        "hold_days":   int(sell_idx - buy_idx) + 1,
        "sell_idx":    int(sell_idx),
    }


# ════════════════════════════════════════════════════════════
#  跑一组参数
# ════════════════════════════════════════════════════════════

def run_param(prep: dict, signal_fn, signal_kwargs: dict,
              start: str, end: str) -> pd.DataFrame:
    """
    对整个股票池跑一组参数。

    signal_fn(p: dict, **signal_kwargs) -> np.ndarray[bool]（与该标的等长，True=当日收盘买入信号）
        p 是 prepare_universe 的单标的 dict（含 close/high/low/volume/... 数组）

    返回: trades DataFrame（仅 signal_date ∈ [start, end] 的交易；同标的持仓不重叠）
    """
    trades: list[dict] = []
    for sym, p in prep.items():
        n = p["n"]
        dates = p["dates"]
        in_range = (dates >= start) & (dates <= end)
        if not in_range.any():
            continue

        sig = signal_fn(p, **signal_kwargs)
        if sig is None:
            continue
        sig = np.asarray(sig, dtype=bool)
        if sig.shape[0] != n:
            raise ValueError(f"signal_fn 返回长度 {sig.shape[0]} != {n}（symbol={sym}）")

        # 静态可交易性掩码（与参数无关，能预筛）
        amt_ma20 = p["amt_ma20"]
        close    = p["close"]
        pct      = p["pct"]
        lp       = p["limit_pct"]
        idx_ok   = np.zeros(n, dtype=bool)
        if n > MIN_DATA_DAYS:
            idx_ok[MIN_DATA_DAYS:] = True
        liquid_ok = (~np.isnan(amt_ma20)) & (amt_ma20 >= MIN_AMOUNT_W * 1e4)
        price_ok  = close >= MIN_PRICE
        notlimit  = np.abs(pct) < (lp - 0.1)          # 与 limit_rules.is_limit_move 一致
        eligible  = sig & in_range & idx_ok & liquid_ok & price_ok & notlimit

        cand = np.where(eligible)[0]
        if cand.size == 0:
            continue

        p = dict(p); p["_sym"] = sym                  # 浅拷贝塞个 symbol，不污染原 prep
        next_free = 0
        for i in cand:
            if i < next_free:                         # 还在上一笔持仓里
                continue
            tr = _simulate_trade(p, int(i))
            if tr is None:
                continue
            trades.append(tr)
            next_free = tr["sell_idx"] + 1

    return pd.DataFrame(trades)


# ════════════════════════════════════════════════════════════
#  单槽顺序组合（参考用的"可实现收益曲线"）
# ════════════════════════════════════════════════════════════

def single_slot_portfolio(trades: pd.DataFrame, init_cash: float = 1.0) -> dict:
    """
    把全市场交易按 buy_date 排序后贪心地"单槽"复利：空仓时拿下一个能拿的信号，
    持到出场再拿下一个。给出一条可比较的资金曲线（不受信号频率膨胀影响）。

    返回: {n_slot, total_ret_pct, max_dd_pct, ann_sharpe, span_days}
      total_ret_pct : 复利总收益 %
      max_dd_pct    : 资金曲线最大回撤 %（负数）
      ann_sharpe    : 用每笔收益序列粗略年化的夏普（√(年交易笔数) 缩放）
    """
    if trades is None or len(trades) == 0:
        return {"n_slot": 0, "total_ret_pct": 0.0, "max_dd_pct": 0.0, "ann_sharpe": 0.0, "span_days": 0}

    t = trades.sort_values(["buy_date", "sell_date"]).reset_index(drop=True)
    cash = init_cash
    equity_path = []          # 每笔后的资金
    rets = []
    last_sell = ""            # 上一笔的出场日（字符串可直接比较）；空仓位必须 buy_date > last_sell 才能再进
    n_slot = 0
    for _, r in t.iterrows():
        if r["buy_date"] <= last_sell:       # 还在持仓中（含同日），跳过这个信号
            continue
        r_pct = float(r["net_pct"]) / 100.0
        cash *= (1.0 + r_pct)
        rets.append(r_pct)
        equity_path.append(cash)
        last_sell = r["sell_date"]
        n_slot += 1

    if n_slot == 0:
        return {"n_slot": 0, "total_ret_pct": 0.0, "max_dd_pct": 0.0, "ann_sharpe": 0.0, "span_days": 0}

    eq = pd.Series(equity_path)
    max_dd = float(((eq - eq.cummax()) / eq.cummax()).min()) * 100.0
    total_ret = (cash / init_cash - 1.0) * 100.0

    # 粗略年化夏普：用第一笔买入日到最后一笔卖出日的跨度推算"年均交易笔数"
    d0 = datetime.strptime(t.iloc[0]["buy_date"], "%Y-%m-%d")
    d1 = datetime.strptime(t.iloc[-1]["sell_date"], "%Y-%m-%d")
    span_days = max((d1 - d0).days, 1)
    trades_per_year = n_slot / (span_days / 365.0)
    r = pd.Series(rets)
    sd = float(r.std())
    ann_sharpe = (float(r.mean()) / sd * np.sqrt(trades_per_year)) if (sd and sd > 1e-9) else 0.0

    return {
        "n_slot":        n_slot,
        "total_ret_pct": round(total_ret, 1),
        "max_dd_pct":    round(max_dd, 1),
        "ann_sharpe":    round(ann_sharpe, 3),
        "span_days":     span_days,
    }


# ════════════════════════════════════════════════════════════
#  汇总指标
# ════════════════════════════════════════════════════════════

def summarize_trades(trades: pd.DataFrame) -> dict:
    """
    每组参数的评价指标。

    交易级（衡量"信号质量"，样本大、不随信号频率膨胀）：
      n / win_rate / avg_ret / med_ret / pl_ratio / avg_hold
      sharpe_t : 交易级夏普 = mean(net_pct)/std(net_pct)（无量纲，跨参数可比）
      sum_ret  : 所有交易 net_pct 之和（"累计捕获的 edge"，会随 n 膨胀，仅参考）

    组合级（衡量"单槽复利下能实现多少"，参考）：
      port_ret  : single_slot_portfolio 的复利总收益 %
      port_maxdd: 该资金曲线最大回撤 %
      port_sharpe / n_slot
    """
    if trades is None or len(trades) == 0:
        return {"n": 0, "win_rate": 0.0, "avg_ret": 0.0, "med_ret": 0.0, "sum_ret": 0.0,
                "sharpe_t": 0.0, "pl_ratio": None, "avg_hold": 0.0,
                "port_ret": 0.0, "port_maxdd": 0.0, "port_sharpe": 0.0, "n_slot": 0}

    t = trades.sort_values("buy_date").reset_index(drop=True)
    n = len(t)
    wins = int(t["win"].sum())
    avg = float(t["net_pct"].mean())
    med = float(t["net_pct"].median())
    sum_ret = float(t["net_pct"].sum())
    std = float(t["net_pct"].std())
    sharpe_t = (avg / std) if (std and std > 1e-9) else 0.0
    win_ret  = t.loc[t["net_pct"] > 0, "net_pct"].mean()
    loss_ret = t.loc[t["net_pct"] <= 0, "net_pct"].mean()
    if loss_ret is not None and loss_ret == loss_ret and loss_ret != 0:
        pl_ratio = abs(float(win_ret) / float(loss_ret))
    else:
        pl_ratio = None

    port = single_slot_portfolio(t)

    return {
        "n":           n,
        "win_rate":    round(wins / n * 100, 1),
        "avg_ret":     round(avg, 3),
        "med_ret":     round(med, 3),
        "sum_ret":     round(sum_ret, 1),
        "sharpe_t":    round(sharpe_t, 3),
        "pl_ratio":    round(pl_ratio, 2) if pl_ratio is not None else None,
        "avg_hold":    round(float(t["hold_days"].mean()), 1),
        "port_ret":    port["total_ret_pct"],
        "port_maxdd":  port["max_dd_pct"],
        "port_sharpe": port["ann_sharpe"],
        "n_slot":      port["n_slot"],
    }


# ── 硬筛阈值（任务1 决策4：复合指标 = 先硬筛、再按夏普排序）──
HARD_MIN_TRADES    = 30        # 交易级样本量下限（你的 <30 不下结论纪律）
HARD_MIN_WINRATE   = 45.0      # 胜率下限 %
HARD_MAX_PORT_DD   = -25.0     # 单槽组合资金曲线最大回撤上限 %（A 股给到 25%）


def passes_hard_filter(m: dict) -> bool:
    return (m.get("n", 0) >= HARD_MIN_TRADES
            and m.get("win_rate", 0.0) >= HARD_MIN_WINRATE
            and m.get("port_maxdd", -999.0) >= HARD_MAX_PORT_DD)
