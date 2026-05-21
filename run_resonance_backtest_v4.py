"""
run_resonance_backtest_v4.py — 共振策略历史回测 V4（固定 +10% 止盈 复活版）

跟 run_resonance_backtest.py 同流程，只换出场规则。**原版 V5 不动**，方便对比。

出场规则（V4，与你 4 月前那版一致）：
  - T+1 开盘买入（A股T+1规则）
  - 止损：买入价下跌 STOP_LOSS_PCT（-5%）立即出场（盘中 low 触发）
  - 止盈：盘中 high ≥ 买入价 × (1+FIXED_TP_PCT)（+10%）出场，成交价记为目标价
  - 到期：持满 MAX_HOLD_DAYS（20 天）强制收盘出场

为什么用 V4：
  研究里 14 个出场变体对比（research/param_search/grid_exits.py）+ canonical 验证
  （research/param_search/run_resonance_canonical_exits.py）一致显示 V4 最 robust：
    - 样本内 2024-01~2025-09（1395 信号）   ：sh_t 第 1 (0.151)，均+1.05%/笔
    - 样本外 2025-10~2026-03（474 信号）    ：sh_t 第 1 (0.288)，均+1.97%/笔
    - canonical cap=100（同 101 信号对比）  ：sh_t 第 3 (0.472)，均+2.99%/笔，sum_ret **+301.7%**
  V5（移动止盈）在三段样本全垫底。

共振规则保持不变：boll_rv 必须命中 + 至少一个趋势策略（macd / ma_trend）
扫描逻辑、市场过滤、策略参数全部沿用原版默认。

CSV 输出：resonance_backtest_result_v4.csv（跟原版 v5 的 csv 分开放，方便横向对比）
"""

import os
import sys
import sqlite3
from collections import defaultdict
from datetime import datetime

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DB_PATH
import run_resonance_backtest as RB     # 复用原版的所有 helper（scan/market_filter/resonance）

# ── 区间 & 出场参数（V4） ──
START_DATE       = RB.START_DATE         # 跟原版默认对齐：2025-10-01
END_DATE         = RB.END_DATE           #                  2026-03-31
TARGET_SIGNALS   = RB.TARGET_SIGNALS     # 100
RESONANCE_WINDOW = RB.RESONANCE_WINDOW   # 3 日窗口

STOP_LOSS_PCT    = -0.05                 # 硬止损：买入价 -5%（与 V5 一致，定死）
FIXED_TP_PCT     =  0.10                 # 固定止盈：+10%（V4 的核心，区别于 V5 移动止盈）
MAX_HOLD_DAYS    = 20                    # 持满 20 个交易日强制出场

OUTPUT_CSV       = "resonance_backtest_result_v4.csv"

# 信号缓存：跟 research/param_search/run_resonance_canonical_exits.py 共用
# 同 (start, end, target) 第二次跑就是秒回（扫描原本要 ~18 分钟）
SIG_CACHE_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "research", "param_search", "out")
SIG_CACHE_PATH   = os.path.join(SIG_CACHE_DIR,
    f"canonical_signals_{START_DATE}_{END_DATE}_n{TARGET_SIGNALS}.csv")


# ════════════════════════════════════════════════════════════
#  V4 出场（同接口替换 RB.simulate_trade）
# ════════════════════════════════════════════════════════════

def simulate_trade_v4(conn, symbol: str, signal_date: str) -> dict | None:
    """
    V4 出场：
      1. T+1 开盘买入
      2. 每日 high >= 买入×(1+10%) → 固定止盈成交在目标价
      3. 每日 low  <= 买入×(1-5%)  → 止损成交在止损价
      4. 持满 MAX_HOLD_DAYS → 当日收盘出场

    ⚠️ V4 不维护"持仓最高价回撤"的 trailing；高点 peak_price 仍记录便于报告。
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
    buy_price = float(future.iloc[0]["open"])
    if buy_price <= 0:
        return None

    target_price = buy_price * (1.0 + FIXED_TP_PCT)
    stop_price   = buy_price * (1.0 + STOP_LOSS_PCT)

    peak_price  = buy_price
    sell_date   = None
    sell_price  = None
    exit_reason = "到期"

    for idx in range(len(future)):
        row = future.iloc[idx]

        if row["high"] > peak_price:
            peak_price = float(row["high"])

        # 止损（盘中 low 触发，成交在 stop_price）
        if row["low"] <= stop_price:
            sell_date, sell_price, exit_reason = row["trade_date"], stop_price, "止损"
            break

        # 固定 +10% 止盈（盘中 high 触发，成交在 target_price）
        if row["high"] >= target_price:
            sell_date, sell_price, exit_reason = row["trade_date"], target_price, "固定止盈"
            break

        # 到期
        if idx >= MAX_HOLD_DAYS - 1:
            sell_date, sell_price, exit_reason = row["trade_date"], float(row["close"]), "到期"
            break

    if sell_date is None:
        last = future.iloc[-1]
        sell_date, sell_price, exit_reason = last["trade_date"], float(last["close"]), "数据截止"

    net_pct  = (sell_price - buy_price) / buy_price * 100
    peak_pct = (peak_price - buy_price) / buy_price * 100

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


# ════════════════════════════════════════════════════════════
#  主流程（结构跟原版 main 几乎一样，但 simulate 换成 V4 + 输出标签调成 V4）
# ════════════════════════════════════════════════════════════

def main():
    print(f"\n{'='*60}")
    print(f"  共振策略历史回测 V4（固定 +10% 止盈，复活版）")
    print(f"  共振规则：boll_rv ∩ (macd 或 ma_trend)")
    print(f"  出场规则：止损{STOP_LOSS_PCT*100:.0f}% / 固定止盈+{FIXED_TP_PCT*100:.0f}% / 持满 {MAX_HOLD_DAYS} 天")
    print(f"  扫描区间：{START_DATE} ~ {END_DATE}")
    print(f"  目标信号：{TARGET_SIGNALS} 条")
    print(f"{'='*60}\n")

    conn = sqlite3.connect(DB_PATH)

    try:
        info_df = pd.read_sql("SELECT symbol, name FROM stock_info ORDER BY symbol", conn)
        all_symbols = info_df["symbol"].tolist()
        name_map = dict(zip(
            info_df["symbol"].astype(str).str.zfill(6),
            info_df["name"].fillna(""),
        ))
    except Exception:
        all_symbols = pd.read_sql("SELECT DISTINCT symbol FROM daily_bars", conn)["symbol"].tolist()
        name_map = {}

    print(f"股票池：{len(all_symbols)} 只")
    trade_dates = RB.get_all_trade_dates(conn, START_DATE, END_DATE)
    print(f"交易日数：{len(trade_dates)} 天\n")

    if not trade_dates:
        print("❌ 无有效交易日")
        conn.close()
        return

    # ── 第一阶段：扫描（直接复用原版 RB 的扫描函数；优先用缓存） ──
    print("── 第一阶段：扫描历史共振信号 ──")
    scanned_dates = 0
    skipped_weak  = 0

    if os.path.exists(SIG_CACHE_PATH):
        print(f"  ✅ 从缓存读 canonical 信号: {SIG_CACHE_PATH}")
        cached = pd.read_csv(SIG_CACHE_PATH, dtype={"symbol": str})
        signals_found = cached.to_dict("records")
        print(f"  共振信号 {len(signals_found)} 条（来自缓存，跳过 SQL 扫描）")
    else:
        signals_found = []
        seen_symbols  = set()
        date_cache    = {}
        breadth_cache = {}

        early_stopped = False
        for idx, td in enumerate(trade_dates):
            ok, reason = RB.is_market_ok(conn, td, breadth_cache)
            if not ok:
                skipped_weak += 1
                print(f"  [{idx+1:3d}/{len(trade_dates)}] {td}  ⛔ {reason}", end="\r")
                continue

            scanned_dates += 1
            if td not in date_cache:
                date_cache[td] = RB.scan_one_date(conn, all_symbols, td, name_map)

            window_dates = RB.get_recent_dates_before(conn, td, RESONANCE_WINDOW)
            window_hits  = [date_cache[d] for d in window_dates if d in date_cache]
            resonance    = RB.find_resonance_symbols(window_hits)

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
                early_stopped = True
                break
        if not early_stopped:
            print(f"\n  扫描完毕，共 {len(signals_found)} 条")

        # 把扫到的信号写入缓存（同 START/END/TARGET 下次秒回）
        try:
            os.makedirs(SIG_CACHE_DIR, exist_ok=True)
            pd.DataFrame(signals_found).to_csv(SIG_CACHE_PATH, index=False, encoding="utf-8-sig")
            print(f"  信号缓存写入 {SIG_CACHE_PATH}")
        except Exception as e:
            print(f"  ⚠️ 缓存写入失败：{e}")

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

    # ── 第二阶段：V4 模拟交易 ──
    print("── 第二阶段：V4 模拟交易（固定 +10% 止盈） ──")
    trades = []
    skip_count = 0

    for i, sig in enumerate(signals_found):
        result = simulate_trade_v4(conn, sig["symbol"], sig["signal_date"])
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
    print(f"  V4 回测结果汇总（共 {total} 笔交易）")
    print(f"{'='*60}")
    print(f"  胜率           : {win_rate:.1f}%  ({int(wins)}胜 / {total-int(wins)}负)")
    print(f"  平均收益       : {avg_ret:+.2f}%")
    print(f"  平均持仓最高点 : {avg_peak:+.2f}%")
    print(f"  最大单笔盈利   : {df['net_pct'].max():+.2f}%")
    print(f"  最大单笔亏损   : {df['net_pct'].min():+.2f}%")
    print(f"  跳过（数据不足）: {skip_count} 笔")
    print(f"  出场方式       : "
          f"止损={exit_cnt.get('止损',0)}  "
          f"固定止盈={exit_cnt.get('固定止盈',0)}  "
          f"到期={exit_cnt.get('到期',0)}")
    print(f"{'='*60}")
    print(f"  累计收益（等权）: {df_sorted['cum_return'].iloc[-1]:+.2f}%")
    print(f"  最大回撤（等权）: {drawdown:.2f}%")
    print(f"{'='*60}\n")

    df_sorted.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"  详细记录已保存：{OUTPUT_CSV}\n")

    print("  各出场方式收益分布：")
    for reason, group in df.groupby("exit_reason"):
        print(f"    {reason:8s}  共{len(group):3d}笔  "
              f"胜率{group['win'].mean()*100:.0f}%  "
              f"均收益{group['net_pct'].mean():+.2f}%  "
              f"均最高{group['peak_pct'].mean():+.2f}%")

    print(f"\n{'='*60}")
    print(f"  V4 vs V5 横向参考（同区间，cap={TARGET_SIGNALS}）")
    print(f"{'='*60}")
    print(f"  研究层 canonical 对比（research/param_search/run_resonance_canonical_exits.py）")
    print(f"  V4_fix10  : 胜率 62.4%, 均+2.99%, sum +301.7%, PL 1.74, 持仓 13.9d")
    print(f"  V5_trail5 : 胜率 62.4%, 均+1.17%, sum +118.5%, PL 1.35, 持仓 12.3d")
    print(f"  → 同样 101 条信号，V4 累计收益是 V5 的 ~2.5 倍")
    print(f"  本次实跑：胜率 {win_rate:.1f}%, 均{avg_ret:+.2f}%, sum {df_sorted['cum_return'].iloc[-1]:+.2f}%, 持仓"
          f" {(pd.to_datetime(df['sell_date'])-pd.to_datetime(df['buy_date'])).dt.days.mean():.1f}d")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    start_time = datetime.now()
    main()
    elapsed = (datetime.now() - start_time).seconds
    print(f"总耗时：{elapsed//60}分{elapsed%60}秒")
