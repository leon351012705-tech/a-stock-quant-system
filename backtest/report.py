"""
【回测层】report.py — 回测报告输出

生成：
  - 控制台文字报告
  - 资金曲线图（对比基准）
  - 交易记录明细
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")   # 非交互模式，避免 Windows 弹窗问题
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backtest.metrics import calc_metrics, print_metrics

# 解决 matplotlib 中文显示问题
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


def generate_report(
    result: dict,
    strategy_name: str = "策略",
    symbol: str = "",
    output_dir: str = None,
    show_trades: bool = True,
) -> dict:
    """
    生成完整回测报告。

    参数：
        result        : Backtester.run() 的返回值
        strategy_name : 策略名称，用于标题和文件名
        symbol        : 股票代码
        output_dir    : 图表保存目录，None则保存到项目 logs 目录
        show_trades   : 是否打印交易明细

    返回：
        metrics dict
    """
    metrics = calc_metrics(result)
    title = f"{strategy_name}" + (f"（{symbol}）" if symbol else "")

    # ── 1. 打印指标 ──
    print_metrics(metrics, title)

    # ── 2. 打印交易明细 ──
    trades_df = result["trades"]
    if show_trades and not trades_df.empty:
        print(f"\n  交易明细（共 {len(trades_df)} 笔）：")
        print("  " + "─" * 70)
        cols = [c for c in ["date", "action", "price", "shares", "commission", "pnl", "pnl_pct"]
                if c in trades_df.columns]
        col_names = {"date": "日期", "action": "操作", "price": "价格",
                     "shares": "股数", "commission": "手续费", "pnl": "盈亏(元)", "pnl_pct": "盈亏(%)"}
        print_df = trades_df[cols].copy()
        print_df.columns = [col_names.get(c, c) for c in cols]
        print_df["日期"] = print_df["日期"].astype(str).str[:10]
        for col in ["价格", "手续费", "盈亏(元)", "盈亏(%)"]:
            if col in print_df.columns:
                print_df[col] = print_df[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "")
        print(print_df.to_string(index=False))

    # ── 3. 绘制资金曲线 ──
    _plot_equity_curve(result, title, output_dir)

    return metrics


def _plot_equity_curve(result: dict, title: str, output_dir: str = None):
    """绘制资金曲线图并保存。"""
    equity_df = result["equity_curve"]
    trades_df = result["trades"]
    initial   = result["initial_cash"]

    # 归一化收益率曲线
    equity_norm = equity_df["total_equity"] / initial * 100 - 100

    fig, axes = plt.subplots(2, 1, figsize=(12, 8),
                             gridspec_kw={"height_ratios": [3, 1]})
    fig.suptitle(f"回测报告：{title}", fontsize=14, fontweight="bold")

    # ── 上图：资金曲线 ──
    ax1 = axes[0]
    dates = equity_df["date"]
    ax1.plot(dates, equity_norm, color="#2196F3", linewidth=1.5, label="策略收益率(%)")
    ax1.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax1.fill_between(dates, equity_norm, 0,
                     where=(equity_norm >= 0), alpha=0.1, color="#2196F3")
    ax1.fill_between(dates, equity_norm, 0,
                     where=(equity_norm < 0),  alpha=0.1, color="#F44336")

    # 标记买卖点
    if not trades_df.empty:
        buys  = trades_df[trades_df["action"] == "买入"]
        sells = trades_df[trades_df["action"] == "卖出"]

        for _, row in buys.iterrows():
            idx = equity_df[equity_df["date"] == row["date"]].index
            if len(idx) > 0:
                y = equity_norm.iloc[idx[0]]
                ax1.scatter(row["date"], y, color="#F44336", marker="^",
                            s=80, zorder=5)

        for _, row in sells.iterrows():
            idx = equity_df[equity_df["date"] == row["date"]].index
            if len(idx) > 0:
                y = equity_norm.iloc[idx[0]]
                color = "#2196F3" if row.get("pnl", 0) >= 0 else "#F44336"
                ax1.scatter(row["date"], y, color=color, marker="v",
                            s=80, zorder=5)

    ax1.set_ylabel("累计收益率 (%)")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    # ── 下图：回撤 ──
    ax2 = axes[1]
    equity = equity_df["total_equity"]
    rolling_max = equity.cummax()
    drawdown = (equity - rolling_max) / rolling_max * 100
    ax2.fill_between(dates, drawdown, 0, color="#F44336", alpha=0.4, label="回撤(%)")
    ax2.set_ylabel("回撤 (%)")
    ax2.legend(loc="lower left")
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    plt.tight_layout()

    # 保存图表
    if output_dir is None:
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(base, "logs")
    os.makedirs(output_dir, exist_ok=True)

    safe_title = title.replace("/", "_").replace(" ", "_")
    filepath = os.path.join(output_dir, f"backtest_{safe_title}.png")
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  📊 资金曲线图已保存：{filepath}")
