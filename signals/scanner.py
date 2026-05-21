"""
【信号层】signals/scanner.py — 多策略全市场扫描

共振规则（V4）：
  boll_rv 必须命中 + 至少一个趋势策略（macd / ma_trend）

缓存机制：
  市场环境不可做时，仍完成扫描并写入缓存，只跳过信号推送。
  这样共振池的 3 日窗口不会因为某天被过滤而出现缓存缺口。

多维 ranker（signals/ranker.py）:
  共振池里每只票额外打分（0~1）：信号新鲜度 + 共振强度 + 趋势对齐 + 流动性 + 风险扣分。
"""

import pandas as pd
import sqlite3
import requests
import logging
import os
import sys
import time
import smtplib
from email.mime.text import MIMEText
from email.utils import formataddr
from datetime import datetime, date, timedelta
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DB_PATH, SERVERCHAN_KEY
from data.limit_rules import get_limit_pct, is_limit_move
from signals.ranker import calc_signal_score, score_tag

import research.strategies.macd_cross as macd_strategy
import research.strategies.boll_band as boll_strategy
import research.strategies.ma_trend as ma_strategy
import research.strategies.ssb_bounce as ssb_strategy

logger = logging.getLogger(__name__)

# ── 扫描参数 ──
MIN_DATA_DAYS    = 60         # 最少历史天数，不足则跳过
MIN_AMOUNT       = 5000       # 最低 20 日均成交额（**单位：万元**），< 此值视为低流动性跳过
LOOKBACK_DAYS    = 120        # 每只股加载的历史窗口（用于策略计算）
RESONANCE_WINDOW = 3          # 共振窗口（交易日）

# ── 策略分类 ──
TREND_STRATEGIES    = {"macd", "ma_trend"}
REVERSAL_STRATEGIES = {"boll_rv", "ssb"}

STRATEGIES = [
    {
        "id": "macd",
        "name": "MACD金叉",
        "func": lambda df: macd_strategy.generate_signals(df, zero_axis_filter=False),
        "max_signals": 20,
        "score_func": None,
    },
    {
        "id": "boll_rv",
        "name": "布林带超跌反弹",
        "func": lambda df: boll_strategy.generate_signals(df, mode="reversion"),
        "max_signals": 20,
        "score_func": None,
    },
    {
        "id": "ma_trend",
        "name": "均线多头排列",
        "func": lambda df: ma_strategy.generate_signals(df, use_entry_filter=True),
        "max_signals": 20,
        "score_func": None,
    },
    {
        "id": "ssb",
        "name": "SSB趋势回踩",
        "func": lambda df: ssb_strategy.generate_signals(df),
        "max_signals": 5,
        "score_func": ssb_strategy.get_signal_score,
    },
]


# ════════════════════════════════════════
#  缓存层
# ════════════════════════════════════════

def _ensure_cache_table(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS signal_cache (
            trade_date  TEXT NOT NULL,
            strategy_id TEXT NOT NULL,
            symbol      TEXT NOT NULL,
            close       REAL,
            pct_change  REAL,
            amount_w    REAL,
            score       REAL,
            PRIMARY KEY (trade_date, strategy_id, symbol)
        )
    """)
    conn.commit()


def _is_date_cached(conn, trade_date: str) -> bool:
    row = conn.execute(
        "SELECT COUNT(*) FROM signal_cache WHERE trade_date = ?", (trade_date,)
    ).fetchone()
    return row[0] > 0


def _write_cache(conn, trade_date: str, raw_results: dict):
    rows = []
    for sid, result_list in raw_results.items():
        for r in result_list:
            rows.append((
                trade_date, sid, r["symbol"],
                r.get("close", 0), r.get("pct_change", 0),
                r.get("amount_w", 0), r.get("score", 0),
            ))
    conn.executemany("""
        INSERT OR REPLACE INTO signal_cache
            (trade_date, strategy_id, symbol, close, pct_change, amount_w, score)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, rows)
    conn.commit()
    logger.info("[缓存] 已写入 %d 条记录（%s）", len(rows), trade_date)


def _read_cache(conn, trade_date: str) -> dict:
    df = pd.read_sql(
        "SELECT strategy_id, symbol FROM signal_cache WHERE trade_date = ?",
        conn, params=(trade_date,)
    )
    raw_hits = defaultdict(set)
    for _, row in df.iterrows():
        raw_hits[row["strategy_id"]].add(row["symbol"])
    return raw_hits


# ════════════════════════════════════════
#  共振判断（V4：boll_rv 必须命中）
# ════════════════════════════════════════

def _is_valid_resonance(hit_strategies: set) -> bool:
    return "boll_rv" in hit_strategies and bool(hit_strategies & TREND_STRATEGIES)


# ════════════════════════════════════════
#  数据新鲜度检查
# ════════════════════════════════════════

def _expected_latest_workday() -> str:
    """
    今天或之前最近的开市日。
    优先用 trade_calendar 表（识别法定节假日）；
    日历空时回落到周一-周五硬判断（节后第一天会假警告）。
    """
    today_str = date.today().strftime("%Y-%m-%d")

    # 优先用日历
    try:
        from data.trade_calendar import latest_trading_day_on_or_before
        result = latest_trading_day_on_or_before(today_str)
        if result:
            return result
    except Exception:
        pass

    # 回落：周一-周五
    d = date.today()
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d.strftime("%Y-%m-%d")


def _check_data_freshness(latest_date: str) -> None:
    """对比 DB 最新日期和"应有最新日期"，过时则打印醒目警告。"""
    expected = _expected_latest_workday()
    if latest_date >= expected:
        return  # 数据已是最新

    today_dow = ["一", "二", "三", "四", "五", "六", "日"][date.today().weekday()]
    now = datetime.now()

    logger.warning("=" * 60)
    logger.warning("⚠️  数据新鲜度警告")
    logger.warning("    数据库最新交易日 : %s", latest_date)
    logger.warning("    应有最新交易日   : %s（今日周%s）", expected, today_dow)
    if date.today().weekday() < 5 and now.hour < 17:
        logger.warning("    可能原因         : 今日盘后数据尚未发布（建议 17:30 后再跑）")
    elif date.today().weekday() < 5:
        logger.warning("    可能原因         : 法定节假日，或日常更新尚未运行")
    else:
        logger.warning("    可能原因         : 今日为周末")
    logger.warning("    本次扫描将使用   : %s 的数据（不含今日盘面动能）", latest_date)
    logger.warning("    建议             : 先跑 python run_daily_update_v2.py")
    logger.warning("=" * 60)


# ════════════════════════════════════════
#  宽松共振池
# ════════════════════════════════════════

def _get_recent_trade_dates(conn, latest_date: str, n: int) -> list:
    rows = pd.read_sql(
        """
        SELECT DISTINCT trade_date FROM daily_bars
        WHERE trade_date <= ?
        ORDER BY trade_date DESC
        LIMIT ?
        """,
        conn, params=(latest_date, n),
    )["trade_date"].tolist()
    return sorted(rows)


def _build_resonance_pool(conn, latest_date: str, strategies: list,
                          grouped: dict | None = None,
                          name_map: dict | None = None) -> pd.DataFrame:
    """
    grouped: {symbol: DataFrame(LOOKBACK_DAYS rows asc)} — 用于打分
    name_map: {symbol: name} — 用于 ST 识别
    None 时退化为旧行为（按 [hit_count, today_hit, amount_w] 排序）
    """
    trade_dates = _get_recent_trade_dates(conn, latest_date, RESONANCE_WINDOW)
    logger.info("[共振池] 窗口：%s", trade_dates)

    missing = [td for td in trade_dates if not _is_date_cached(conn, td)]
    if missing:
        logger.warning("[共振池] 以下日期无缓存，共振可能不完整：%s", missing)

    sym_strategy_map = defaultdict(set)
    sym_date_map     = defaultdict(set)

    for td in trade_dates:
        if not _is_date_cached(conn, td):
            continue
        day_hits = _read_cache(conn, td)
        for sid, sym_set in day_hits.items():
            for sym in sym_set:
                sym_strategy_map[sym].add(sid)
                sym_date_map[sym].add(td)

    candidates = [
        sym for sym, strats in sym_strategy_map.items()
        if _is_valid_resonance(strats)
    ]

    if not candidates:
        return pd.DataFrame()

    # 一次性查所有候选股的当日行情（旧实现：每只 1 次 SELECT）
    placeholders = ",".join(["?"] * len(candidates))
    today_df = pd.read_sql(
        f"""
        SELECT symbol, close, pct_change, amount FROM daily_bars
        WHERE trade_date = ? AND symbol IN ({placeholders})
        """,
        conn, params=[latest_date, *candidates],
    )
    today_lookup = {
        row["symbol"]: row for _, row in today_df.iterrows()
    }

    today_info = {}
    for sym in candidates:
        row = today_lookup.get(sym)
        if row is not None:
            today_info[sym] = {
                "close":     row["close"],
                "pct_change":row["pct_change"],
                "amount_w":  round(row["amount"] / 10000, 1),
                "today_has_data": 1,   # 今日有数据（非停牌）
            }
        else:
            today_info[sym] = {"close": 0, "pct_change": 0, "amount_w": 0, "today_has_data": 0}

    rows = []
    for sym in candidates:
        strats    = sorted(sym_strategy_map[sym])
        hit_dates_sym = sorted(sym_date_map[sym])
        info      = today_info[sym]

        # 多维打分（如果上层传入了行情数据）
        score = 0.0
        score_info = {}
        tag = ""
        if grouped is not None:
            df_sym = grouped.get(sym)
            if df_sym is not None and not df_sym.empty:
                # grouped 来自 _scan_market，已按 trade_date 升序
                df_sym_sorted = df_sym.reset_index(drop=True)
                stock_name = (name_map or {}).get(sym, "")
                score, score_info = calc_signal_score(
                    df_sym_sorted, set(strats), set(hit_dates_sym),
                    stock_name, latest_date,
                )
                tag = score_tag(score, score_info)

        rows.append({
            "symbol":         sym,
            "trade_date":     latest_date,
            "close":          info["close"],
            "pct_change":     info["pct_change"],
            "amount_w":       info["amount_w"],
            "hit_count":      len(strats),
            "hit_strategies": ",".join(strats),
            "hit_dates":      ",".join(hit_dates_sym),
            "today_has_data": info["today_has_data"],   # 今日有数据（非停牌）
            "score":          score,
            "tag":            tag,
            "gain_60d":       score_info.get("gain_60d", 0.0),
        })

    df_res = pd.DataFrame(rows)

    # 排序：有 score 时按 score 降序；无（旧行为）时按原 hit_count
    if grouped is not None and (df_res["score"] > 0).any():
        df_res = df_res.sort_values(
            by=["score", "today_has_data", "amount_w"],
            ascending=[False, False, False],
        ).head(5).reset_index(drop=True)
    else:
        df_res = df_res.sort_values(
            by=["hit_count", "today_has_data", "amount_w"],
            ascending=[False, False, False],
        ).head(5).reset_index(drop=True)

    return df_res


# ════════════════════════════════════════
#  全市场扫描（抽出为独立函数）
# ════════════════════════════════════════

def _scan_market(conn, symbols: list, latest_date: str, strategies: list) -> tuple[dict, dict, dict, dict]:
    """
    扫描全市场，返回 (raw_results, stats)
    raw_results: {strategy_id: [row_info, ...]}
    无论市场好坏都可调用，结果用于写缓存。

    实现：一次查全市场 LOOKBACK_DAYS 天数据，groupby 在内存分组。
    旧实现是 5499 次单股 SELECT，I/O 主导；新实现 1 次 SELECT，CPU 主导。
    """
    raw_results = {s["id"]: [] for s in strategies}
    stats = {
        "total_symbols":  len(symbols),
        "skip_short_data":  0,
        "skip_suspended":   0,
        "skip_low_amount":  0,
        "skip_limit_move":  0,
        "passed_filters":   0,
        "strategy_runs":    {s["id"]: 0 for s in strategies},
        "strategy_hits":    {s["id"]: 0 for s in strategies},
        "strategy_errors":  {s["id"]: 0 for s in strategies},
    }

    # 取最近 LOOKBACK_DAYS 个交易日，用最早那天作为查询下界
    cutoff_dates = pd.read_sql(
        """
        SELECT DISTINCT trade_date FROM daily_bars
        WHERE trade_date <= ? ORDER BY trade_date DESC LIMIT ?
        """,
        conn, params=(latest_date, LOOKBACK_DAYS),
    )["trade_date"].tolist()

    if not cutoff_dates:
        logger.warning("[扫描] daily_bars 表为空")
        return raw_results, stats

    cutoff = min(cutoff_dates)
    logger.info("[扫描] 批量加载 %s ~ %s 全市场数据 ...", cutoff, latest_date)

    t0 = time.time()
    all_df = pd.read_sql(
        """
        SELECT symbol, trade_date, open, high, low, close, volume, amount, pct_change, turnover
        FROM daily_bars
        WHERE trade_date >= ? AND trade_date <= ?
        ORDER BY symbol, trade_date
        """,
        conn, params=(cutoff, latest_date),
    )
    logger.info("[扫描] 加载完成：%d 行 / %d 股，用时 %.1fs",
                len(all_df), all_df["symbol"].nunique(), time.time() - t0)

    # 按 symbol 分组（已 ORDER BY symbol, trade_date，组内自动按日期升序）
    grouped = dict(tuple(all_df.groupby("symbol", sort=False)))

    # 一次性拿全部股票名字（用于 ST 识别）
    name_df = pd.read_sql("SELECT symbol, name FROM stock_info", conn)
    name_map = dict(zip(name_df["symbol"].astype(str).str.zfill(6), name_df["name"].fillna("")))

    for i, symbol in enumerate(symbols):
        if (i + 1) % 1000 == 0:
            logger.info("进度 %d/%d ...", i + 1, len(symbols))

        df = grouped.get(symbol)
        if df is None or len(df) < MIN_DATA_DAYS:
            stats["skip_short_data"] += 1
            continue

        # 已经 ORDER BY 升序，直接 reset_index 即可（不再 sort_values）
        df = df.reset_index(drop=True)

        if df["trade_date"].iloc[-1] != latest_date:
            stats["skip_suspended"] += 1
            continue
        recent_amount = df["amount"].tail(20).mean()
        if recent_amount < MIN_AMOUNT * 10000:
            stats["skip_low_amount"] += 1
            continue

        # 按板块/ST 算正确的涨跌停阈值
        limit_pct = get_limit_pct(symbol, name_map.get(symbol, ""), latest_date)
        if is_limit_move(df["pct_change"].iloc[-1], limit_pct):
            stats["skip_limit_move"] += 1
            continue

        # 把 limit_pct 透传给策略（ssb_bounce 内部还要做一次涨跌停过滤）
        df.attrs["limit_pct"] = limit_pct
        df.attrs["symbol"] = symbol

        stats["passed_filters"] += 1

        # 今日成交额（与共振池语义统一）
        today_amount = float(df["amount"].iloc[-1])

        for strategy in strategies:
            sid = strategy["id"]
            try:
                stats["strategy_runs"][sid] += 1
                signals = strategy["func"](df)
                if signals.iloc[-1] == 1:
                    stats["strategy_hits"][sid] += 1
                    row_info = {
                        "symbol":     symbol,
                        "close":      df["close"].iloc[-1],
                        "pct_change": df["pct_change"].iloc[-1],
                        "amount_w":   round(today_amount / 10000, 1),   # 今日成交额（与共振池一致）
                        "amount_w_20d_avg": round(recent_amount / 10000, 1),  # 备用：近 20 日均
                        "trade_date": latest_date,
                    }
                    score_func = strategy.get("score_func")
                    if score_func is not None:
                        try:
                            row_info["score"] = score_func(df, len(df) - 1)
                        except Exception:
                            row_info["score"] = 0
                    else:
                        row_info["score"] = 0
                    raw_results[sid].append(row_info)
            except Exception as e:
                stats["strategy_errors"][sid] += 1
                logger.warning("策略[%s] 股票[%s]失败：%s", sid, symbol, e)

    return raw_results, stats, grouped, name_map


# ════════════════════════════════════════
#  主扫描函数
# ════════════════════════════════════════

def run_scan(strategies: list = None, top_n: int = None) -> dict:
    if strategies is None:
        strategies = STRATEGIES

    conn = sqlite3.connect(DB_PATH)
    _ensure_cache_table(conn)

    latest_date = pd.read_sql(
        "SELECT MAX(trade_date) as d FROM daily_bars", conn
    ).iloc[0]["d"]

    if pd.isna(latest_date):
        logger.error("daily_bars 表为空")
        conn.close()
        result = {s["id"]: pd.DataFrame() for s in strategies}
        result["resonance"] = pd.DataFrame()
        return result

    logger.info("扫描基准日期：%s", latest_date)

    # 数据新鲜度检查：DB 落后于"应有最新日"时打印警告
    _check_data_freshness(latest_date)

    # ── 市场环境判断 ──
    is_bullish = True
    try:
        from data.market_filter import get_market_filter
        market_filter = get_market_filter()
        market_status = market_filter.get_market_status(latest_date)

        logger.info("=" * 60)
        logger.info("市场环境检查:")
        logger.info("  交易日: %s",       market_status.get("trade_date", latest_date))
        logger.info("  样本数: %s",       market_status.get("total", "N/A"))
        logger.info("  上涨占比: %s%%",   market_status.get("up_ratio", "N/A"))
        logger.info("  下跌占比: %s%%",   market_status.get("down_ratio", "N/A"))
        logger.info("  大跌股占比: %s%%", market_status.get("big_drop_ratio", "N/A"))
        logger.info("  中位涨跌幅: %s%%", market_status.get("median_pct", "N/A"))
        logger.info("  平均涨跌幅: %s%%", market_status.get("mean_pct", "N/A"))
        logger.info("  判断: %s", "✅ 可做多" if market_status["is_bullish"] else "❌ 不可做")
        logger.info("  原因: %s",         market_status.get("reason", ""))
        logger.info("=" * 60)

        is_bullish = market_status["is_bullish"]

        if not is_bullish:
            logger.warning("⚠️ 市场环境不佳，信号不推送——但仍扫描以维护缓存完整性")

    except Exception as e:
        logger.warning("市场过滤器加载失败：%s", e)

    # ── 获取股票列表 ──
    symbols = pd.read_sql(
        "SELECT symbol FROM stock_info ORDER BY symbol", conn
    )["symbol"].tolist()
    if not symbols:
        symbols = pd.read_sql(
            "SELECT DISTINCT symbol FROM daily_bars", conn
        )["symbol"].tolist()

    logger.info("开始扫描 %d 只股票，策略数：%d", len(symbols), len(strategies))

    # ── 全市场扫描（无论市场好坏都跑，保证缓存完整）──
    raw_results, stats, grouped, name_map = _scan_market(conn, symbols, latest_date, strategies)

    # ── 写入缓存（关键：提前到市场判断之后，信号过滤之前）──
    _write_cache(conn, latest_date, raw_results)

    # ── 市场不好时，返回空结果（不推送信号）──
    if not is_bullish:
        logger.warning("⚠️ 市场环境不佳，跳过信号输出")
        conn.close()
        result = {s["id"]: pd.DataFrame() for s in strategies}
        result["resonance"] = pd.DataFrame()
        return result

    # ── 单策略结果截断 ──
    final = {}
    for s in strategies:
        sid   = s["id"]
        max_n = top_n if top_n is not None else s.get("max_signals", 20)
        if raw_results[sid]:
            df_result = pd.DataFrame(raw_results[sid])
            if s.get("score_func") is not None:
                df_result = df_result.sort_values("score", ascending=False).head(max_n)
            else:
                df_result = df_result.sort_values("amount_w", ascending=False).head(max_n)
            final[sid] = df_result
            logger.info("[%s] 触发信号 %d 只（取前%d）", s["name"], len(raw_results[sid]), max_n)
        else:
            final[sid] = pd.DataFrame()
            logger.info("[%s] 今日无信号", s["name"])

    # ── 共振池 ──
    logger.info("[共振池] 构建中（窗口=%d）...", RESONANCE_WINDOW)
    resonance_df = _build_resonance_pool(conn, latest_date, strategies,
                                         grouped=grouped, name_map=name_map)
    final["resonance"] = resonance_df

    if resonance_df.empty:
        logger.info("[共振池] 今日无信号")
    else:
        logger.info("[共振池] 触发 %d 只", len(resonance_df))
        for _, row in resonance_df.iterrows():
            logger.info(
                "  %s  策略：%s  触发日：%s  成交额：%.0f万",
                row["symbol"], row["hit_strategies"],
                row["hit_dates"], row["amount_w"]
            )

    conn.close()

    logger.info("────────── 扫描统计 ──────────")
    logger.info("股票总数           : %d", stats["total_symbols"])
    logger.info("数据不足跳过       : %d", stats["skip_short_data"])
    logger.info("停牌跳过           : %d", stats["skip_suspended"])
    logger.info("流动性不足跳过     : %d", stats["skip_low_amount"])
    logger.info("涨跌停跳过         : %d", stats["skip_limit_move"])
    logger.info("通过全部过滤       : %d", stats["passed_filters"])
    for s in strategies:
        sid = s["id"]
        logger.info("策略[%s] 运行=%d 命中=%d 报错=%d",
                    s["name"], stats["strategy_runs"][sid],
                    stats["strategy_hits"][sid], stats["strategy_errors"][sid])
    logger.info("──────────────────────────────")

    return final


# ════════════════════════════════════════
#  推送 & 打印
# ════════════════════════════════════════

def _format_amount(amount_w) -> str:
    """成交额（万元）→ 美化字符串：< 1 亿用万，≥ 1 亿用亿"""
    try:
        v = float(amount_w)
    except (TypeError, ValueError):
        return "—"
    if v >= 10000:
        return f"{v/10000:.1f}亿"
    return f"{int(v)}万"


def _batch_get_names(symbols: list) -> dict:
    """一次拿所有名字，避免 N+1 查询"""
    if not symbols:
        return {}
    try:
        conn = sqlite3.connect(DB_PATH)
        placeholders = ",".join(["?"] * len(symbols))
        df = pd.read_sql(
            f"SELECT symbol, name FROM stock_info WHERE symbol IN ({placeholders})",
            conn, params=list(symbols),
        )
        conn.close()
        return dict(zip(df["symbol"], df["name"].fillna("")))
    except Exception:
        return {}


def _build_push_content(scan_results: dict, strategies: list) -> tuple[str, str]:
    """美化版邮件：去除冗余字段，紧凑层次清晰"""
    today = datetime.today()
    weekday_cn = "一二三四五六日"[today.weekday()]
    title = f"📊 量化信号 {today:%Y-%m-%d}"

    SEP_HEAVY = "═" * 32
    SEP_LIGHT = "─" * 32

    # 一次性拿所有要展示股票的名字（避免 N+1 查询）
    all_syms = set()
    for sid in [s["id"] for s in strategies] + ["resonance"]:
        df = scan_results.get(sid, pd.DataFrame())
        if df is not None and not df.empty and "symbol" in df.columns:
            all_syms.update(df["symbol"].astype(str).tolist())
    name_map = _batch_get_names(list(all_syms))
    def nm(sym):
        return name_map.get(sym, sym)

    lines = []
    lines.append(f"📊 量化信号  {today:%Y-%m-%d} 周{weekday_cn}")
    lines.append(SEP_HEAVY)
    lines.append("")

    total_signals = 0

    # ===== 共振池：用户最关心，最显眼 =====
    res_df = scan_results.get("resonance", pd.DataFrame())
    if res_df is not None and not res_df.empty:
        total_signals += len(res_df)
        lines.append(f"🔥 共振池 ({len(res_df)} 只)")
        lines.append(SEP_LIGHT)
        for i, (_, row) in enumerate(res_df.iterrows(), 1):
            sym = row["symbol"]
            pct = row["pct_change"]
            pct_str = f"{pct:+.2f}%" if pd.notna(pct) else "—"
            score = row.get("score", 0)
            tag = row.get("tag", "")
            score_part = f"  ⭐{score:.2f}" if score else ""
            tag_part = f" {tag}" if tag else ""
            # 60 日涨幅：仅显示数字 + 中性图标，不下判断（小样本回测没找到稳定预测力）
            g60 = row.get("gain_60d", 0.0)
            g60_part = f"  60日{g60:+.1f}%"
            lines.append(
                f"{i}. {nm(sym)} {sym}  {row['close']:.2f} {pct_str}{score_part}{tag_part}{g60_part}"
            )
        lines.append("")
    else:
        lines.append("🔥 共振池 · 今日无信号")
        lines.append("")

    # ===== 单策略池 =====
    EMOJI = {"macd": "📈", "boll_rv": "📉", "ma_trend": "📊", "ssb": "🎯"}
    for s in strategies:
        sid = s["id"]
        name = s["name"]
        df = scan_results.get(sid, pd.DataFrame())
        if df is None or df.empty:
            continue
        total_signals += len(df)
        emoji = EMOJI.get(sid, "•")
        lines.append(f"{emoji} {name} ({len(df)} 只)")
        lines.append(SEP_LIGHT)
        has_score = "score" in df.columns and s.get("score_func") is not None
        for _, row in df.iterrows():
            sym = row["symbol"]
            line = f"{nm(sym)} {sym}  {row['close']:.2f} {row['pct_change']:+.2f}%  {_format_amount(row['amount_w'])}"
            if has_score:
                line += f"  ⭐{row['score']:.2f}"
            lines.append(line)
        lines.append("")

    # ===== 页脚 =====
    lines.append(SEP_LIGHT)
    if total_signals == 0:
        lines.append("今日全市场无触发信号，耐心等待")
    else:
        lines.append(f"共 {total_signals} 条信号 · 重点看 🔥 共振池")
    lines.append(SEP_HEAVY)

    return title, "\n".join(lines)


def _push_via_serverchan(title: str, content: str) -> bool:
    """微信推送（Server酱）。成功返回 True"""
    if not SERVERCHAN_KEY or SERVERCHAN_KEY == "YOUR_SERVERCHAN_KEY_HERE":
        return False
    url = f"https://sctapi.ftqq.com/{SERVERCHAN_KEY}.send"
    try:
        resp = requests.post(url, data={"title": title, "desp": content}, timeout=10)
        if resp.json().get("code") == 0:
            logger.info("微信推送成功：%s", title)
            return True
        logger.warning("微信推送返回异常：%s", resp.text)
    except Exception as e:
        logger.error("微信推送失败：%s", e)
    return False


def _push_via_email(title: str, content: str) -> bool:
    """邮件推送（SMTP，用 .env 里的 EMAIL_* 配置）。成功返回 True"""
    host     = os.environ.get("EMAIL_SMTP_HOST", "").strip()
    port     = int(os.environ.get("EMAIL_SMTP_PORT", "465") or "465")
    sender   = os.environ.get("EMAIL_FROM", "").strip()
    password = os.environ.get("EMAIL_PASSWORD", "").strip()
    receiver = os.environ.get("EMAIL_TO", sender).strip()  # 不填默认发给自己

    if not (host and sender and password and receiver):
        return False

    msg = MIMEText(content, "plain", "utf-8")
    msg["From"]    = formataddr(("量化信号", sender))
    msg["To"]      = receiver
    msg["Subject"] = title

    recipients = [r.strip() for r in receiver.split(",") if r.strip()]

    try:
        if port == 465:
            srv = smtplib.SMTP_SSL(host, port, timeout=15)
        else:
            srv = smtplib.SMTP(host, port, timeout=15)
            srv.starttls()
        srv.login(sender, password)
        srv.sendmail(sender, recipients, msg.as_string())
        srv.quit()
        logger.info("邮件推送成功：%s → %s", title, receiver)
        return True
    except Exception as e:
        logger.error("邮件推送失败：%s", e)
        return False


def push_to_wechat(scan_results: dict, strategies: list = None):
    """
    名字保留向后兼容（旧调用方仍能跑），实际可同时推送到：
      - Server酱（SERVERCHAN_KEY 已配置时）
      - 邮箱（EMAIL_* 已配置时）
    没有任何渠道配置时，回落到控制台打印。
    """
    if strategies is None:
        strategies = STRATEGIES

    title, content = _build_push_content(scan_results, strategies)

    pushed = []
    if _push_via_serverchan(title, content):
        pushed.append("微信")
    if _push_via_email(title, content):
        pushed.append("邮箱")

    if not pushed:
        logger.warning("未配置任何推送渠道（Server酱 / 邮箱），改为控制台打印")
        _print_results(scan_results, strategies)
    else:
        logger.info("推送完成：%s", " + ".join(pushed))


def _print_results(scan_results: dict, strategies: list = None):
    if strategies is None:
        strategies = STRATEGIES
    today = datetime.today().strftime("%Y-%m-%d")
    print(f"\n📊 量化信号 {today}")
    print("=" * 60)

    resonance_df = scan_results.get("resonance", pd.DataFrame())
    print(f"\n【🔥 趋势+反转双确认共振（近{RESONANCE_WINDOW}日）】")
    if resonance_df is None or resonance_df.empty:
        print("  今日无信号")
    else:
        for _, row in resonance_df.iterrows():
            pct_str = f"{row['pct_change']:+.2f}%" if row["pct_change"] != 0 else "N/A"
            print(f"  {_get_stock_name(row['symbol'])}（{row['symbol']}）"
                  f"  收盘 {row['close']:.2f}  涨跌 {pct_str}"
                  f"  命中 [{row['hit_strategies']}]"
                  f"  触发日 [{row['hit_dates']}]"
                  f"  成交额 {row['amount_w']:.0f}万")

    for s in strategies:
        df = scan_results.get(s["id"], pd.DataFrame())
        has_score = "score" in df.columns and s.get("score_func") is not None
        print(f"\n【{s['name']}】")
        if df.empty:
            print("  今日无信号")
        else:
            for _, row in df.iterrows():
                line = (f"  {_get_stock_name(row['symbol'])}（{row['symbol']}）"
                        f"  收盘 {row['close']:.2f}"
                        f"  涨跌 {row['pct_change']:+.2f}%"
                        f"  成交额 {row['amount_w']:.0f}万")
                if has_score:
                    line += f"  评分 {row['score']:.2f}"
                print(line)
    print("=" * 60)


def _get_stock_name(symbol: str) -> str:
    try:
        conn = sqlite3.connect(DB_PATH)
        result = pd.read_sql(
            "SELECT name FROM stock_info WHERE symbol = ?",
            conn, params=(symbol,),
        )
        conn.close()
        return result.iloc[0]["name"] if not result.empty else symbol
    except Exception:
        return symbol
