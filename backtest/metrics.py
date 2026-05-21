"""
【回测层】metrics.py — 绩效指标计算

计算回测结果的所有关键指标，与主流平台对齐：
  - 年化收益率
  - 最大回撤
  - 夏普比率
  - 胜率 / 盈亏比
  - 平均持仓天数
"""

import pandas as pd
import numpy as np


def calc_metrics(result: dict) -> dict:
    """
    根据回测结果计算全套绩效指标。

    参数：
        result : Backtester.run() 的返回值

    返回：
        指标 dict
    """
    equity_df = result["equity_curve"]
    trades_df = result["trades"]
    initial   = result["initial_cash"]
    final     = result["final_equity"]

    # ── 基本收益 ──
    total_return = (final - initial) / initial * 100

    # ── 年化收益率 ──
    days = (equity_df["date"].iloc[-1] - equity_df["date"].iloc[0]).days
    years = days / 365
    annual_return = ((final / initial) ** (1 / years) - 1) * 100 if years > 0 else 0

    # ── 最大回撤 ──
    equity = equity_df["total_equity"]
    rolling_max = equity.cummax()
    drawdown = (equity - rolling_max) / rolling_max * 100
    max_drawdown = drawdown.min()

    # ── 夏普比率（年化，无风险利率取 2.5%）──
    daily_returns = equity.pct_change().dropna()
    rf_daily = 0.025 / 252
    excess_returns = daily_returns - rf_daily

    std = excess_returns.std()
    if std is not None and np.isfinite(std) and std > 1e-8:
        sharpe = excess_returns.mean() / std * np.sqrt(252)
    else:
        sharpe = 0

    # ── 交易统计 ──
    if not trades_df.empty:
        sell_trades = trades_df[trades_df["action"] == "卖出"].copy()
        buy_trades  = trades_df[trades_df["action"] == "买入"].copy()
    else:
        sell_trades = pd.DataFrame()
        buy_trades  = pd.DataFrame()

    total_trades = len(sell_trades)

    if total_trades > 0 and "pnl" in sell_trades.columns:
        wins   = sell_trades[sell_trades["pnl"] > 0]
        losses = sell_trades[sell_trades["pnl"] <= 0]

        win_rate = len(wins) / total_trades * 100

        avg_win  = wins["pnl"].mean()   if len(wins) > 0 else 0
        avg_loss = losses["pnl"].mean() if len(losses) > 0 else 0

        if avg_loss != 0:
            profit_loss_ratio = abs(avg_win / avg_loss)
        else:
            profit_loss_ratio = None  # 无亏损样本时记为 None，展示时输出 ∞

        total_pnl = sell_trades["pnl"].sum()
    else:
        win_rate = 0
        profit_loss_ratio = None
        total_pnl = 0

    # ── 平均持仓天数 ──
    avg_hold_days = 0
    if not buy_trades.empty and not sell_trades.empty:
        buy_dates  = buy_trades["date"].reset_index(drop=True)
        sell_dates = sell_trades["date"].reset_index(drop=True)
        n = min(len(buy_dates), len(sell_dates))

        if n > 0:
            hold_days = [(sell_dates[j] - buy_dates[j]).days for j in range(n)]
            avg_hold_days = np.mean(hold_days)

    # ── 总交易成本（佣金 + 印花税）──
    if not trades_df.empty:
        commission_sum = trades_df["commission"].sum() if "commission" in trades_df.columns else 0
        stamp_duty_sum = trades_df["stamp_duty"].sum() if "stamp_duty" in trades_df.columns else 0
        total_commission = commission_sum + stamp_duty_sum
    else:
        total_commission = 0

    return {
        "总收益率(%)"   : round(total_return, 2),
        "年化收益率(%)" : round(annual_return, 2),
        "最大回撤(%)"   : round(max_drawdown, 2),
        "夏普比率"      : round(sharpe, 3),
        "交易次数"      : total_trades,
        "胜率(%)"       : round(win_rate, 2),
        "盈亏比"        : round(profit_loss_ratio, 2) if profit_loss_ratio is not None else "∞",
        "总盈亏(元)"    : round(total_pnl, 2),
        "总手续费(元)"  : round(total_commission, 2),
        "平均持仓  数"  : round(avg_hold_days, 1),
        "回测天数"      : days,
        "期初资金(元)"  : initial,
        "期末资金(元)"  : round(final, 2),
    }


def print_metrics(metrics: dict, title: str = "回测结果"):
    """格式化打印绩效指标。"""
    print("\n" + "═" * 40)
    print(f"  {title}")
    print("═" * 40)
    for k, v in metrics.items():
        print(f"  {k:<14} {v}")
    print("═" * 40)