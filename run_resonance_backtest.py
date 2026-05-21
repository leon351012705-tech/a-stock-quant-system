"""
run_resonance_backtest.py
共振策略历史回测 V5 — 移动止盈（跟踪最高价回撤）

出场规则：
  - 止损：买入价下跌 STOP_LOSS_PCT（-5%）立即出场
  - 移动止盈：持仓期间记录最高价，从最高价回撤 TRAIL_PCT（-5%）出场
    （不设固定止盈上限，让趋势跑多远就拿多远）
  - 到期：持满 MAX_HOLD_DAYS 天强制出场

共振规则：boll_rv 必须命中 + 至少一个趋势策略（macd / ma_trend）
"""

import os
import sys
import sqlite3
import pandas as pd
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DB_PATH
from data.limit_rules import get_limit_pct, is_limit_move

import research.strategies.macd_cross as macd_strategy
import research.strategies.boll_band  as boll_strategy
import research.strategies.ma_trend   as ma_strategy
import research.strategies.ssb_bounce as ssb_strategy

# ── 回测参数 ──
START_DATE       = "2025-10-01"
END_DATE         = "2026-03-31"
TARGET_SIGNALS   = 100
RESONANCE_WINDOW = 3
MAX_HOLD_DAYS    = 20             # 移动止盈需要更长窗口，从10改到20
STOP_LOSS_PCT    = -0.05          # 止损：买入价 -5%
TRAIL_PCT        = -0.05          # 移动止盈：最高价回撤 -5% 出场
MIN_DATA_DAYS    = 60
MIN_AMOUNT       = 5000
LOOKBACK_DAYS    = 120

# ── 策略分类 ──
TREND_STRATEGIES = {"macd", "ma_trend"}

STRATEGIES = [
    {"id": "macd",     "func": lambda df: macd_strategy.generate_signals(df, zero_axis_filter=False)},
    {"id": "boll_rv",  "func": lambda df: boll_strategy.generate_signals(df, mode="reversion")},
    {"id": "ma_trend", "func": lambda df: ma_strategy.generate_signals(df, use_entry_filter=True)},
    {"id": "ssb",      "func": lambda df: ssb_strategy.generate_signals(df)},
]


# ════════════════════════════════════════
#  市场过滤
# ════════════════════════════════════════

UP_RATIO_MIN   = 0.45
UP_RATIO_WEAK  = 0.40
BIG_DROP_MAX   = 0.20
MEDIAN_PCT_MIN = -0.5


def get_breadth(conn, trade_date: str) -> dict | None:
    df = pd.read_sql(
        "SELECT pct_change FROM daily_bars WHERE trade_date = ? AND pct_change IS NOT NULL",
        conn, params=(trade_date,),
    )
    if len(df) < 100:
        return None
    total = len(df)
    return {
        "up_ratio":       (df["pct_change"] > 0).sum() / total,
        "big_drop_ratio": (df["pct_change"] < -4).sum() / total,
        "median_pct":     df["pct_change"].median(),
    }


def is_market_ok(conn, trade_date: str, breadth_cache: dict) -> tuple[bool, str]:
    if trade_date not in breadth_cache:
        breadth_cache[trade_date] = get_breadth(conn, trade_date)
    b = breadth_cache[trade_date]
    if b is None:
        return False, "当日数据不足"
    day_ok = (
        (b["up_ratio"] >= UP_RATIO_MIN and b["big_drop_ratio"] <= BIG_DROP_MAX)
        or
        (b["up_ratio"] >= UP_RATIO_WEAK and b["median_pct"] >= MEDIAN_PCT_MIN)
    )
    return (True, "广度OK") if day_ok else (False, f"广度偏弱（上涨{b['up_ratio']*100:.1f}%）")


# ════════════════════════════════════════
#  共振判断
# ════════════════════════════════════════

def is_valid_resonance(hit_strategies: set) -> bool:
    return "boll_rv" in hit_strategies and bool(hit_strategies & TREND_STRATEGIES)


def find_resonance_symbols(window_hits: list) -> list:
    sym_strats = defaultdict(set)
    for day_hits in window_hits:
        for sid, sym_set in day_hits.items():
            for sym in sym_set:
                sym_strats[sym].add(sid)
    return [sym for sym, strats in sym_strats.items() if is_valid_resonance(strats)]


# ════════════════════════════════════════
#  扫描工具
# ════════════════════════════════════════

def scan_one_date(conn, all_symbols: list, target_date: str,
                  name_map: dict | None = None) -> dict:
    raw_hits = {s["id"]: set() for s in STRATEGIES}
    if name_map is None:
        name_map = {}
    for symbol in all_symbols:
        df = pd.read_sql(
            """
            SELECT trade_date, open, high, low, close, volume, amount, pct_change, turnover
            FROM daily_bars
            WHERE symbol = ? AND trade_date <= ?
            ORDER BY trade_date DESC LIMIT ?
            """,
            conn, params=(symbol, target_date, LOOKBACK_DAYS),
        )
        if len(df) < MIN_DATA_DAYS:
            continue
        df = df.sort_values("trade_date").reset_index(drop=True)
        if df["trade_date"].iloc[-1] != target_date:
            continue
        if df["amount"].tail(20).mean() < MIN_AMOUNT * 10000:
            continue

        # 板块/ST 感知的涨跌停
        limit_pct = get_limit_pct(symbol, name_map.get(symbol, ""), target_date)
        if is_limit_move(df["pct_change"].iloc[-1], limit_pct):
            continue

        df.attrs["limit_pct"] = limit_pct
        df.attrs["symbol"] = symbol

        for s in STRATEGIES:
            try:
                signals = s["func"](df)
                if signals.iloc[-1] == 1:
                    raw_hits[s["id"]].add(symbol)
            except Exception as e:
                # 不再静默吞错——至少记下来便于调试
                import logging
                logging.getLogger(__name__).warning(
                    "[backtest] 策略 %s 处理 %s 异常：%s", s["id"], symbol, e
                )
    return raw_hits


def get_all_trade_dates(conn, start: str, end: str) -> list:
    return pd.read_sql(
        """
        SELECT DISTINCT trade_date FROM daily_bars
        WHERE trade_date >= ? AND trade_date <= ?
        ORDER BY trade_date
        """,
        conn, params=(start, end),
    )["trade_date"].tolist()


def get_recent_dates_before(conn, before_date: str, n: int) -> list:
    return sorted(pd.read_sql(
        """
        SELECT DISTINCT trade_date FROM daily_bars
        WHERE trade_date <= ? ORDER BY trade_date DESC LIMIT ?
        """,
        conn, params=(before_date, n),
    )["trade_date"].tolist())


# ════════════════════════════════════════
#  模拟交易（移动止盈）
# ════════════════════════════════════════

def simulate_trade(conn, symbol: str, signal_date: str) -> dict | None:
    """
    出场逻辑：
      1. T+1 开盘买入（A股T+1规则）
      2. 每日更新最高价（用当日 high）
      3. 当日收盘价 or 当日最低价触发止损/移动止盈时出场
      4. 持满 MAX_HOLD_DAYS 强制出场
    """
    future = pd.read_sql(
        """
        SELECT trade_date, open, high, low, close
        FROM daily_bars
        WHERE symbol = ? AND trade_date > ?
        ORDER BY trade_date LIMIT ?
        """,
        conn, params=(symbol, signal_date, MAX_HOLD_DAYS + 1),
    )

    if len(future) < 2:
        return None

    buy_date  = future.iloc[0]["trade_date"]
    buy_price = future.iloc[0]["open"]
    if buy_price <= 0:
        return None

    peak_price   = buy_price    # 持仓期间最高价
    sell_date    = None
    sell_price   = None
    exit_reason  = "到期"
    peak_date    = buy_date

    for idx in range(len(future)):
        row = future.iloc[idx]

        # 更新最高价（用当日最高价）
        if row["high"] > peak_price:
            peak_price = row["high"]
            peak_date  = row["trade_date"]

        # 止损检查：最低价触发
        stop_price  = buy_price * (1 + STOP_LOSS_PCT)
        if row["low"] <= stop_price:
            sell_date, sell_price, exit_reason = row["trade_date"], stop_price, "止损"
            break

        # 移动止盈检查：从最高价回撤 TRAIL_PCT
        trail_price = peak_price * (1 + TRAIL_PCT)
        if row["low"] <= trail_price and peak_price > buy_price * 1.01:
            # 只有最高价比买入价高1%以上才触发移动止盈（防止刚买就小幅波动）
            sell_date, sell_price, exit_reason = row["trade_date"], trail_price, "移动止盈"
            break

        # 到期强制出场
        if idx >= MAX_HOLD_DAYS - 1:
            sell_date, sell_price, exit_reason = row["trade_date"], row["close"], "到期"
            break

    if sell_date is None:
        last = future.iloc[-1]
        sell_date, sell_price, exit_reason = last["trade_date"], last["close"], "数据截止"

    net_pct   = (sell_price - buy_price) / buy_price * 100
    peak_pct  = (peak_price - buy_price) / buy_price * 100

    return {
        "symbol":      symbol,
        "signal_date": signal_date,
        "buy_date":    buy_date,
        "buy_price":   round(buy_price, 3),
        "peak_price":  round(peak_price, 3),
        "peak_pct":    round(peak_pct, 2),
        "sell_date":   sell_date,
        "sell_price":  round(sell_price, 3),
        "net_pct":     round(net_pct, 2),
        "exit_reason": exit_reason,
        "win":         1 if net_pct > 0 else 0,
    }


# ════════════════════════════════════════
#  主流程
# ════════════════════════════════════════

def main():
    print(f"\n{'='*60}")
    print(f"  共振策略历史回测 V5（移动止盈）")
    print(f"  共振规则：boll_rv ∩ (macd 或 ma_trend)")
    print(f"  出场规则：止损{STOP_LOSS_PCT*100:.0f}% / 移动止盈（最高价回撤{abs(TRAIL_PCT)*100:.0f}%）/ 持满{MAX_HOLD_DAYS}天")
    print(f"  扫描区间：{START_DATE} ~ {END_DATE}")
    print(f"  目标信号：{TARGET_SIGNALS} 条")
    print(f"{'='*60}\n")

    conn = sqlite3.connect(DB_PATH)

    try:
        info_df = pd.read_sql(
            "SELECT symbol, name FROM stock_info ORDER BY symbol", conn
        )
        all_symbols = info_df["symbol"].tolist()
        name_map = dict(zip(
            info_df["symbol"].astype(str).str.zfill(6),
            info_df["name"].fillna(""),
        ))
    except Exception:
        all_symbols = pd.read_sql(
            "SELECT DISTINCT symbol FROM daily_bars", conn
        )["symbol"].tolist()
        name_map = {}

    print(f"股票池：{len(all_symbols)} 只")
    trade_dates = get_all_trade_dates(conn, START_DATE, END_DATE)
    print(f"交易日数：{len(trade_dates)} 天\n")

    if not trade_dates:
        print("❌ 无有效交易日")
        conn.close()
        return

    # ── 第一阶段：扫描 ──
    print("── 第一阶段：扫描历史共振信号 ──")
    signals_found = []
    seen_symbols  = set()
    date_cache    = {}
    breadth_cache = {}
    skipped_weak  = 0
    scanned_dates = 0

    for idx, td in enumerate(trade_dates):
        ok, reason = is_market_ok(conn, td, breadth_cache)
        if not ok:
            skipped_weak += 1
            print(f"  [{idx+1:3d}/{len(trade_dates)}] {td}  ⛔ {reason}", end="\r")
            continue

        scanned_dates += 1
        if td not in date_cache:
            date_cache[td] = scan_one_date(conn, all_symbols, td, name_map)

        window_dates = get_recent_dates_before(conn, td, RESONANCE_WINDOW)
        window_hits  = [date_cache[d] for d in window_dates if d in date_cache]
        resonance    = find_resonance_symbols(window_hits)

        for sym in resonance:
            if sym not in seen_symbols:
                seen_symbols.add(sym)
                sym_strats = set()
                for wh in window_hits:
                    for sid, sym_set in wh.items():
                        if sym in sym_set:
                            sym_strats.add(sid)
                signals_found.append({
                    "signal_date":    td,
                    "symbol":         sym,
                    "hit_strategies": ",".join(sorted(sym_strats)),
                })

        print(
            f"  [{idx+1:3d}/{len(trade_dates)}] {td}  ✅  "
            f"共振候选:{len(resonance):3d}  累计:{len(signals_found):4d}",
            end="\r"
        )

        if len(signals_found) >= TARGET_SIGNALS:
            print(f"\n  ✅ 已收集满 {TARGET_SIGNALS} 条，停止扫描")
            break
    else:
        print(f"\n  扫描完毕，共 {len(signals_found)} 条")

    print(f"\n── 扫描统计 ──")
    print(f"  实际扫描日期  : {scanned_dates} 天")
    print(f"  广度过滤跳过  : {skipped_weak} 天")
    print(f"  收集信号总数  : {len(signals_found)} 条")

    if signals_found:
        combo_count = defaultdict(int)
        for s in signals_found:
            combo_count[s["hit_strategies"]] += 1
        print(f"\n  命中策略组合：")
        for combo, cnt in sorted(combo_count.items(), key=lambda x: -x[1]):
            print(f"    {combo:<30s}  {cnt} 条")
    print()

    if not signals_found:
        print("❌ 未找到共振信号")
        conn.close()
        return

    # ── 第二阶段：模拟交易 ──
    print("── 第二阶段：模拟交易 ──")
    trades     = []
    skip_count = 0

    for i, sig in enumerate(signals_found):
        result = simulate_trade(conn, sig["symbol"], sig["signal_date"])
        if result is None:
            skip_count += 1
            continue
        result["hit_strategies"] = sig["hit_strategies"]
        trades.append(result)
        print(
            f"  [{i+1:3d}] {sig['symbol']}  "
            f"信号日:{sig['signal_date']}  "
            f"买入:{result['buy_price']}  "
            f"最高:{result['peak_price']}(+{result['peak_pct']:.1f}%)  "
            f"收益:{result['net_pct']:+.2f}%  [{result['exit_reason']}]"
        )

    conn.close()

    if not trades:
        print("❌ 无法模拟")
        return

    # ── 第三阶段：汇总 ──
    df       = pd.DataFrame(trades)
    total    = len(df)
    wins     = df["win"].sum()
    win_rate = wins / total * 100
    avg_ret  = df["net_pct"].mean()
    avg_peak = df["peak_pct"].mean()
    exit_cnt = df["exit_reason"].value_counts().to_dict()

    df_sorted = df.sort_values("buy_date").reset_index(drop=True)
    df_sorted["cum_return"] = df_sorted["net_pct"].cumsum()
    peak     = df_sorted["cum_return"].cummax()
    drawdown = (df_sorted["cum_return"] - peak).min()

    print(f"\n{'='*60}")
    print(f"  回测结果汇总（共 {total} 笔交易）")
    print(f"{'='*60}")
    print(f"  胜率           : {win_rate:.1f}%  ({int(wins)}胜 / {total-int(wins)}负)")
    print(f"  平均收益       : {avg_ret:+.2f}%")
    print(f"  平均持仓最高点 : {avg_peak:+.2f}%  （移动止盈前的平均浮盈）")
    print(f"  最大单笔盈利   : {df['net_pct'].max():+.2f}%")
    print(f"  最大单笔亏损   : {df['net_pct'].min():+.2f}%")
    print(f"  跳过（数据不足）: {skip_count} 笔")
    print(f"  出场方式       : "
          f"止损={exit_cnt.get('止损',0)}  "
          f"移动止盈={exit_cnt.get('移动止盈',0)}  "
          f"到期={exit_cnt.get('到期',0)}")
    print(f"{'='*60}")
    print(f"  累计收益（等权）: {df_sorted['cum_return'].iloc[-1]:+.2f}%")
    print(f"  最大回撤（等权）: {drawdown:.2f}%")
    print(f"{'='*60}\n")

    output_path = "resonance_backtest_result_v5.csv"
    df_sorted.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"  详细记录已保存：{output_path}\n")

    print("  各出场方式收益分布：")
    for reason, group in df.groupby("exit_reason"):
        print(f"    {reason:8s}  共{len(group):3d}笔  "
              f"胜率{group['win'].mean()*100:.0f}%  "
              f"均收益{group['net_pct'].mean():+.2f}%  "
              f"均最高{group['peak_pct'].mean():+.2f}%")

    print(f"\n{'='*60}")
    print(f"  V4 vs V5 对比（相同区间 10月~3月，100笔）")
    print(f"{'='*60}")
    print(f"  {'版本':<20} {'胜率':>8} {'均收益':>10} {'累计收益':>10}")
    print(f"  {'V4 固定止盈10%':<20} {'75.0%':>8} {'+2.96%':>10} {'+295.98%':>10}")
    print(f"  {'V5 移动止盈':<20} {win_rate:>7.1f}% {avg_ret:>+10.2f}% "
          f"{df_sorted['cum_return'].iloc[-1]:>+10.2f}%")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    start_time = datetime.now()
    main()
    elapsed = (datetime.now() - start_time).seconds
    print(f"总耗时：{elapsed//60}分{elapsed%60}秒")
