"""
【回测层】run_batch_backtest.py — 批量回测入口（含基准对比）

运行方式：
    python run_batch_backtest.py

功能：
  - 对一组股票批量执行同一策略回测
  - 输出汇总表
  - 增加买入持有基准收益对比
  - 输出超额收益、是否跑赢基准
  - 可选为每只股票生成详细报告和资金曲线图
"""

import sys
import os
import logging
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

from data.storage import query_daily_bars, get_connection
from backtest.engine import Backtester
from backtest.metrics import calc_metrics
from backtest.report import generate_report

import research.strategies.macd_cross as macd_strategy
import research.strategies.ma_trend   as ma_strategy
import research.strategies.boll_band  as boll_strategy


def _load_name_map() -> dict:
    """一次性拿全部 stock_info 名字（用于 ST 检测）"""
    try:
        conn = get_connection()
        df = pd.read_sql_query("SELECT symbol, name FROM stock_info", conn)
        conn.close()
        return dict(zip(df["symbol"].astype(str).str.zfill(6), df["name"].fillna("")))
    except Exception:
        return {}


STRATEGY_REGISTRY = {
    "macd": {
        "name": "MACD金叉策略",
        "func": lambda df: macd_strategy.generate_signals(df, zero_axis_filter=True),
    },
    "ma_trend": {
        "name": "均线多头排列策略",
        "func": lambda df: ma_strategy.generate_signals(df, use_entry_filter=True),
    },
    "boll_reversion": {
        "name": "布林带均值回归策略",
        "func": lambda df: boll_strategy.generate_signals(df, mode="reversion"),
    },
    "boll_breakout": {
        "name": "布林带趋势突破策略",
        "func": lambda df: boll_strategy.generate_signals(df, mode="breakout"),
    },
}


def calc_buy_and_hold_return(df: pd.DataFrame) -> float:
    """
    计算买入持有收益率（以首日开盘买入、末日收盘卖出近似）。

    返回：
        收益率（%）
    """
    if df.empty or len(df) < 2:
        return 0.0

    first_open = df["open"].iloc[0]
    last_close = df["close"].iloc[-1]

    if first_open <= 0:
        return 0.0

    return (last_close / first_open - 1) * 100


def run_batch_backtest(
    symbols: list,
    strategy_id: str,
    start_date: str = "2020-01-01",
    end_date: str = "2025-12-31",
    initial_cash: float = 100000,
    generate_detail_report: bool = False,
):
    """
    对多个股票批量执行同一策略回测。

    参数：
        symbols                 : 股票代码列表
        strategy_id             : 策略ID
        start_date              : 开始日期
        end_date                : 结束日期
        initial_cash            : 初始资金
        generate_detail_report  : 是否为每只股票生成详细报告
    """
    if strategy_id not in STRATEGY_REGISTRY:
        print(f"✗ 不支持的策略：{strategy_id}")
        print(f"可选策略：{list(STRATEGY_REGISTRY.keys())}")
        return

    strategy_info = STRATEGY_REGISTRY[strategy_id]
    strategy_name = strategy_info["name"]

    print("=" * 100)
    print(f"  批量回测：{strategy_name}")
    print(f"  时间区间：{start_date} ~ {end_date}")
    print(f"  股票数量：{len(symbols)}")
    print("=" * 100)

    summary = []
    name_map = _load_name_map()

    for i, symbol in enumerate(symbols, 1):
        print(f"\n[{i}/{len(symbols)}] 正在回测 {symbol} ...")

        df = query_daily_bars(symbol, start_date=start_date, end_date=end_date)
        if df.empty:
            print(f"  ✗ {symbol} 无数据，跳过")
            continue

        df = df.sort_values("trade_date").reset_index(drop=True)

        try:
            stock_name = name_map.get(str(symbol).zfill(6), "")
            signals = strategy_info["func"](df)
            bt = Backtester(df, initial_cash=initial_cash, symbol=symbol, name=stock_name)
            result = bt.run(signals)
            metrics = calc_metrics(result)

            buy_count  = int((signals == 1).sum())
            sell_count = int((signals == -1).sum())

            benchmark_return = calc_buy_and_hold_return(df)
            excess_return = metrics["总收益率(%)"] - benchmark_return
            beat_benchmark = "是" if excess_return > 0 else "否"

            summary.append({
                "symbol"          : symbol,
                "策略"            : strategy_name,
                "买入信号数"      : buy_count,
                "卖出信号数"      : sell_count,
                "交易次数"        : metrics["交易次数"],
                "策略收益率(%)"   : metrics["总收益率(%)"],
                "基准收益率(%)"   : round(benchmark_return, 2),
                "超额收益(%)"     : round(excess_return, 2),
                "跑赢基准"        : beat_benchmark,
                "年化收益率(%)"   : metrics["年化收益率(%)"],
                "最大回撤(%)"     : metrics["最大回撤(%)"],
                "夏普比率"        : metrics["夏普比率"],
                "胜率(%)"         : metrics["胜率(%)"],
                "盈亏比"          : metrics["盈亏比"],
                "期末资金(元)"    : metrics["期末资金(元)"],
                "期末持仓"        : result.get("open_position", 0),
            })

            print(
                f"  ✓ 完成 | 策略收益 {metrics['总收益率(%)']:.2f}% | "
                f"基准收益 {benchmark_return:.2f}% | "
                f"超额 {excess_return:.2f}% | "
                f"回撤 {metrics['最大回撤(%)']:.2f}%"
            )

            if generate_detail_report:
                generate_report(
                    result,
                    strategy_name=strategy_name,
                    symbol=symbol,
                    show_trades=False,
                )

        except Exception as e:
            print(f"  ✗ {symbol} 回测失败：{e}")
            continue

    if not summary:
        print("\n✗ 没有成功完成任何回测")
        return

    summary_df = pd.DataFrame(summary)
    summary_df = summary_df.sort_values("超额收益(%)", ascending=False).reset_index(drop=True)

    print("\n" + "═" * 120)
    print("  批量回测汇总（含基准对比）")
    print("═" * 120)
    print(summary_df.to_string(index=False))
    print("═" * 120)

    # 汇总统计
    avg_strategy_return = summary_df["策略收益率(%)"].mean()
    avg_benchmark_return = summary_df["基准收益率(%)"].mean()
    avg_excess_return = summary_df["超额收益(%)"].mean()
    avg_drawdown = summary_df["最大回撤(%)"].mean()
    avg_sharpe = summary_df["夏普比率"].mean()
    beat_count = (summary_df["跑赢基准"] == "是").sum()

    print("\n【组合层面统计】")
    print(f"  平均策略收益率 : {avg_strategy_return:.2f}%")
    print(f"  平均基准收益率 : {avg_benchmark_return:.2f}%")
    print(f"  平均超额收益   : {avg_excess_return:.2f}%")
    print(f"  平均最大回撤   : {avg_drawdown:.2f}%")
    print(f"  平均夏普比率   : {avg_sharpe:.3f}")
    print(f"  跑赢基准数量   : {beat_count} / {len(summary_df)}")

    # 保存汇总表
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"batch_backtest_{strategy_id}.csv")
    summary_df.to_csv(filepath, index=False, encoding="utf-8-sig")
    print(f"\n📄 汇总结果已保存：{filepath}")

    return summary_df


def main():
    """
    直接在这里修改要测试的股票池和策略。
    """
    symbols = [
        "600519",  # 贵州茅台
        "000001",  # 平安银行
        "600036",  # 招商银行
        "000333",  # 美的集团
        "601318",  # 中国平安
    ]

    strategy_id = "macd"

    run_batch_backtest(
        symbols=symbols,
        strategy_id=strategy_id,
        start_date="2020-01-01",
        end_date="2025-12-31",
        initial_cash=100000,
        generate_detail_report=False,
    )


if __name__ == "__main__":
    main()