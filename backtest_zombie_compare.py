"""
backtest_zombie_compare.py — 僵尸股过滤层 A/B 回测对比

【核心观察】
B 组（过滤后）的信号集合是 A 组（不过滤）的子集——僵尸股过滤只会
减少信号，不会创造新信号。所以：

  跑一次完整回测 → 对每笔交易标记 was_zombie → 分组对比绩效

无需跑两遍，效率提升 2x。

【配置】
默认小样本快速验证：500 只股票 × 60 个交易日（~5-10 分钟）。
要跑全量改下面的 SAMPLE_SIZE / DATE_RANGE_DAYS。

【输出】
  logs/zombie_ab_trades.csv         — 每笔交易明细（含 was_zombie 标记）
  logs/zombie_ab_summary.csv        — A/B/Zombie 三组绩效对比
  控制台打印对比结论
"""

from __future__ import annotations

import os
import sys
import sqlite3
import random
import logging
import pandas as pd
from collections import defaultdict
from datetime import datetime

# 防 emoji 在 stdout 重定向到文件时炸（Windows 默认 GBK）
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DB_PATH, LOG_DIR
from data.zombie_filter import filter_stock_pool, calculate_zombie_metrics_batch
from zombie_config import ZOMBIE_CONFIG

# 复用现有共振回测的核心函数
from run_resonance_backtest import (
    is_market_ok,
    find_resonance_symbols,
    scan_one_date,
    get_all_trade_dates,
    get_recent_dates_before,
    simulate_trade,
    STRATEGIES,
    RESONANCE_WINDOW,
)

# ════════════════════════════════════════
#  回测配置
# ════════════════════════════════════════

# 中等样本（1 年），用 2025 数据（turnover 完整）
SAMPLE_SIZE = 500            # 抽样股票数（None = 全量）
DATE_RANGE_DAYS = 240        # 回测交易日数（≈ 1 年）
END_DATE = "2025-09-30"
RANDOM_SEED = 42

# 输出
TRADES_CSV  = os.path.join(LOG_DIR, "zombie_ab_trades.csv")
SUMMARY_CSV = os.path.join(LOG_DIR, "zombie_ab_summary.csv")
LOG_FILE    = os.path.join(LOG_DIR, "zombie_filter.log")

# 日志：同时写文件和控制台
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


# ════════════════════════════════════════
#  绩效计算
# ════════════════════════════════════════

def calc_group_metrics(trades_df: pd.DataFrame) -> dict:
    """
    根据交易明细计算绩效指标。

    交易已含 net_pct（单笔收益率 %）和 win 标记。
    """
    if trades_df.empty:
        return {
            "交易笔数":     0,
            "胜率(%)":      0,
            "平均收益率(%)": 0,
            "累计收益率(%)": 0,
            "最大单笔(%)":   0,
            "最大亏损(%)":   0,
            "盈亏比":       0,
            "平均持仓天数":  0,
        }

    n = len(trades_df)
    win_rate = trades_df["win"].mean() * 100
    avg_pct  = trades_df["net_pct"].mean()
    cum_pct  = trades_df["net_pct"].sum()  # 等权累加（每笔 1 万元，等比）
    max_win  = trades_df["net_pct"].max()
    max_loss = trades_df["net_pct"].min()

    wins = trades_df[trades_df["net_pct"] > 0]["net_pct"]
    losses = trades_df[trades_df["net_pct"] <= 0]["net_pct"]
    pl_ratio = (wins.mean() / abs(losses.mean())) if len(losses) > 0 and losses.mean() != 0 else None

    # 平均持仓天数
    if "buy_date" in trades_df.columns and "sell_date" in trades_df.columns:
        buy_dt = pd.to_datetime(trades_df["buy_date"])
        sell_dt = pd.to_datetime(trades_df["sell_date"])
        avg_hold = (sell_dt - buy_dt).dt.days.mean()
    else:
        avg_hold = 0

    return {
        "交易笔数":     n,
        "胜率(%)":      round(win_rate, 2),
        "平均收益率(%)": round(avg_pct, 2),
        "累计收益率(%)": round(cum_pct, 2),
        "最大单笔(%)":   round(max_win, 2),
        "最大亏损(%)":   round(max_loss, 2),
        "盈亏比":       round(pl_ratio, 2) if pl_ratio is not None else "∞",
        "平均持仓天数":  round(avg_hold, 1),
    }


# ════════════════════════════════════════
#  主流程
# ════════════════════════════════════════

def main():
    logger.info("=" * 60)
    logger.info("僵尸股过滤层 A/B 回测对比")
    logger.info("=" * 60)
    logger.info("END_DATE        : %s", END_DATE)
    logger.info("DATE_RANGE_DAYS : %d", DATE_RANGE_DAYS)
    logger.info("SAMPLE_SIZE     : %s", SAMPLE_SIZE if SAMPLE_SIZE else "ALL")
    logger.info("RANDOM_SEED     : %d", RANDOM_SEED)
    logger.info("")

    conn = sqlite3.connect(DB_PATH)

    # ── Step 1: 准备股票池 ──
    info_df = pd.read_sql("SELECT symbol, name FROM stock_info ORDER BY symbol", conn)
    info_df["symbol"] = info_df["symbol"].astype(str).str.zfill(6)
    name_map = dict(zip(info_df["symbol"], info_df["name"].fillna("")))
    all_symbols = info_df["symbol"].tolist()

    # 剔除 ST
    non_st = [s for s in all_symbols if "ST" not in str(name_map.get(s, "")).upper().replace(" ", "")]
    logger.info("全市场 %d 只，剔除 ST 后 %d 只", len(all_symbols), len(non_st))

    # 抽样
    if SAMPLE_SIZE and SAMPLE_SIZE < len(non_st):
        random.seed(RANDOM_SEED)
        sample_symbols = sorted(random.sample(non_st, SAMPLE_SIZE))
    else:
        sample_symbols = non_st
    logger.info("回测样本：%d 只", len(sample_symbols))

    # ── Step 2: 准备日期范围 ──
    trade_dates = pd.read_sql(
        """
        SELECT DISTINCT trade_date FROM daily_bars
        WHERE trade_date <= ? ORDER BY trade_date DESC LIMIT ?
        """,
        conn, params=(END_DATE, DATE_RANGE_DAYS),
    )["trade_date"].tolist()
    trade_dates = sorted(trade_dates)
    logger.info("回测日期：%s ~ %s（%d 天）", trade_dates[0], trade_dates[-1], len(trade_dates))
    logger.info("")

    # ── Step 3: 全市场扫描（只跑一次，得到所有信号）──
    logger.info("【阶段 1/3】全市场扫描 + 共振池构建")
    breadth_cache = {}
    date_cache = {}        # {date: {strategy_id: set(symbols)}}
    zombie_cache = {}      # {date: set(zombie symbols)}
    signals_found = []
    seen_symbols = set()
    skipped_weak = 0

    for idx, td in enumerate(trade_dates):
        # 市场环境过滤
        ok, _ = is_market_ok(conn, td, breadth_cache)
        if not ok:
            skipped_weak += 1
            if (idx + 1) % 10 == 0:
                logger.info("  进度 %d/%d  跳过弱市 %d 天",
                            idx + 1, len(trade_dates), skipped_weak)
            continue

        # 扫描该日（仅一次）
        if td not in date_cache:
            date_cache[td] = scan_one_date(conn, sample_symbols, td, name_map)

        # 算该日的僵尸股集合（用于事后标注）
        if td not in zombie_cache:
            kept = filter_stock_pool(conn, sample_symbols, td, ZOMBIE_CONFIG)
            zombie_cache[td] = set(sample_symbols) - set(kept)

        # 共振信号（A 组：未过滤）
        window_dates = get_recent_dates_before(conn, td, RESONANCE_WINDOW)
        window_hits = [date_cache[d] for d in window_dates if d in date_cache]
        resonance_today = find_resonance_symbols(window_hits)

        for sym in resonance_today:
            if sym in seen_symbols:
                continue
            seen_symbols.add(sym)
            signals_found.append({
                "symbol": sym,
                "signal_date": td,
                "was_zombie": sym in zombie_cache[td],
            })

        if (idx + 1) % 10 == 0:
            logger.info("  进度 %d/%d  累计信号 %d  弱市跳过 %d",
                        idx + 1, len(trade_dates), len(signals_found), skipped_weak)

    logger.info("扫描完成：共 %d 个候选信号", len(signals_found))
    logger.info("  其中 was_zombie=True (B 组会过滤的): %d",
                sum(1 for s in signals_found if s["was_zombie"]))
    logger.info("  其中 was_zombie=False (B 组保留的) : %d",
                sum(1 for s in signals_found if not s["was_zombie"]))
    logger.info("")

    # ── Step 4: 模拟交易 ──
    logger.info("【阶段 2/3】模拟交易")
    trades = []
    for i, sig in enumerate(signals_found):
        result = simulate_trade(conn, sig["symbol"], sig["signal_date"])
        if result is None:
            continue
        result["was_zombie"] = sig["was_zombie"]
        result["name"] = name_map.get(sig["symbol"], "")
        trades.append(result)

        if (i + 1) % 50 == 0:
            logger.info("  交易进度 %d/%d", i + 1, len(signals_found))

    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        logger.warning("⚠️  无可模拟交易，调整时间窗口或样本")
        conn.close()
        return

    logger.info("有效交易：%d 笔", len(trades_df))
    logger.info("")

    # ── Step 5: 三组绩效对比 ──
    logger.info("【阶段 3/3】绩效对比")

    group_all      = trades_df                                           # A 组：不过滤
    group_kept     = trades_df[~trades_df["was_zombie"]]                 # B 组：过滤后保留
    group_filtered = trades_df[trades_df["was_zombie"]]                  # 被过滤的（应该差）

    metrics_all      = calc_group_metrics(group_all)
    metrics_kept     = calc_group_metrics(group_kept)
    metrics_filtered = calc_group_metrics(group_filtered)

    # 输出对比表
    summary = pd.DataFrame({
        "A 组（不过滤·全部）":   metrics_all,
        "B 组（过滤后保留）":    metrics_kept,
        "被过滤组（剔除的）":    metrics_filtered,
    })

    # ── 先保存 CSV（防止后续 print 异常导致数据丢失）──
    trades_df.to_csv(TRADES_CSV, index=False, encoding="utf-8-sig")
    summary.to_csv(SUMMARY_CSV, encoding="utf-8-sig")
    logger.info("交易明细已保存：%s", TRADES_CSV)
    logger.info("绩效对比已保存：%s", SUMMARY_CSV)

    print()
    print("=" * 70)
    print(f"  绩效对比（{trade_dates[0]} ~ {trade_dates[-1]}, {len(sample_symbols)} 只样本）")
    print("=" * 70)
    print(summary.to_string())
    print("=" * 70)

    # ── 关键结论 ──
    delta_winrate = metrics_kept["胜率(%)"] - metrics_all["胜率(%)"]
    delta_avg     = metrics_kept["平均收益率(%)"] - metrics_all["平均收益率(%)"]
    delta_cum     = metrics_kept["累计收益率(%)"] - metrics_all["累计收益率(%)"]

    print()
    print("[结论] B 组 vs A 组:")
    print(f"  胜率差     : {delta_winrate:+.2f}%  {'[提升]' if delta_winrate > 0 else '[下降]' if delta_winrate < 0 else '[持平]'}")
    print(f"  平均收益差 : {delta_avg:+.2f}%  {'[提升]' if delta_avg > 0 else '[下降]' if delta_avg < 0 else '[持平]'}")
    print(f"  累计收益差 : {delta_cum:+.2f}%  {'[提升]' if delta_cum > 0 else '[下降]' if delta_cum < 0 else '[持平]'}")
    print()

    if not group_filtered.empty:
        f_win = metrics_filtered["胜率(%)"]
        f_avg = metrics_filtered["平均收益率(%)"]
        print(f"[被过滤组] 胜率 {f_win}% / 平均收益 {f_avg}%")
        if f_win < metrics_kept["胜率(%)"] and f_avg < metrics_kept["平均收益率(%)"]:
            print("  [验证] 过滤掉的票确实表现更差，过滤层有效")
        else:
            print("  [警告] 过滤掉的票表现不一定差，需要调参或重新审视")
    print()
    logger.info("绩效对比已保存：%s", SUMMARY_CSV)

    conn.close()


if __name__ == "__main__":
    main()
