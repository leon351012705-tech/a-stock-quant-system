"""
data/zombie_filter.py — 僵尸股过滤层

⚠️ 【DEPRECATED — 不在主流程使用】⚠️
2026-05 A/B 回测验证（500 只 × 240 天，47 笔交易）：
  - A 组（不过滤）累计收益 +56.76%，胜率 53.19%
  - B 组（过滤后）累计收益 +29.47%，胜率 58.06%
结论：过滤虽提升胜率 5%，但累计收益砍半——根本原因是设计错误：
  用「过去 60 天没波动」判定僵尸，但策略本身就是寻找「过去没动现在要动」的股票，
  导致"觉醒慢牛"（如 600236 +18%）被误杀。
保留此文件作为研究工具（统计长期沉默股），但 run_signal_scan.py 不调用。
详见 backtest_zombie_compare.py 验证报告。

【目标】（原始设计）
从全市场股票池中剔除"僵尸股"——低波动、低活跃度、长期没行情的股票，
让策略层只对有弹性的标的发信号，提升信号质量和实战回报。

【判定维度（v1 用 4 个 DB 已有数据可算的）】
  1. 60 日振幅 = (60 日最高 - 60 日最低) / 60 日均价
  2. 60 日均换手率（注：DB 在 2026 年起部分缺失，开启容错时跳过）
  3. 60 日均成交额
  4. 近 1 年涨停次数（按板块/ST 涨停阈值精确判定）

满足 ≥ N 个条件则判为僵尸股（默认 N=2）。

【觉醒识别】
即便满足僵尸条件，当日出现以下任一则不过滤：
  - 量比 > 5
  - 实换手 > 5%
  - 涨幅 > 5%

【性能设计】
单次调用 batch query 全市场近 240 天数据，pandas groupby 一次算完。
避免逐股 N+1 查询。

【主入口】
  filter_stock_pool(conn, symbols, end_date, config) → 过滤后的 symbols list
"""

from __future__ import annotations

import logging
import pandas as pd
from data.limit_rules import get_limit_pct

logger = logging.getLogger(__name__)


# ════════════════════════════════════════
#  内部工具
# ════════════════════════════════════════

def _load_window_data(conn, end_date: str, lookback_days: int) -> pd.DataFrame:
    """
    一次查全市场近 lookback_days 个交易日数据。
    返回 DataFrame：symbol, trade_date, open, high, low, close, volume, amount, pct_change, turnover
    """
    # 先找最早的目标交易日（向前推 lookback_days 个交易日）
    dates = pd.read_sql(
        """
        SELECT DISTINCT trade_date FROM daily_bars
        WHERE trade_date <= ? ORDER BY trade_date DESC LIMIT ?
        """,
        conn, params=(end_date, lookback_days),
    )["trade_date"].tolist()

    if not dates:
        return pd.DataFrame()

    start_date = min(dates)

    df = pd.read_sql(
        """
        SELECT symbol, trade_date, open, high, low, close, volume, amount, pct_change, turnover
        FROM daily_bars
        WHERE trade_date >= ? AND trade_date <= ?
        ORDER BY symbol, trade_date
        """,
        conn, params=(start_date, end_date),
    )
    return df


def _get_name_map(conn) -> dict:
    """获取 symbol → name 映射（用于板块/ST 涨停阈值识别）"""
    df = pd.read_sql("SELECT symbol, name FROM stock_info", conn)
    return dict(zip(df["symbol"].astype(str).str.zfill(6),
                    df["name"].fillna("")))


# ════════════════════════════════════════
#  指标计算（单股版本，便于单元测试）
# ════════════════════════════════════════

def calculate_zombie_metrics(stock_code: str, end_date: str, conn=None,
                             config: dict = None) -> dict:
    """
    单股版本：计算指定股票在 end_date 当日的僵尸股特征指标。
    主要用于单元测试和调试，**实际批量场景请用 calculate_zombie_metrics_batch**。

    返回 dict（找不到数据时返回空 dict）：
        symbol, end_date,
        amplitude_60d, turnover_60d_avg, amount_60d_avg,
        limit_up_count_1y,
        today_volume_ratio, today_turnover, today_pct_change,
        data_complete (bool, 是否所有维度都有数据)
    """
    from data.storage import get_connection
    from zombie_config import ZOMBIE_CONFIG

    if config is None:
        config = ZOMBIE_CONFIG
    own_conn = conn is None
    if own_conn:
        conn = get_connection()

    try:
        df = pd.read_sql(
            """
            SELECT trade_date, open, high, low, close, volume, amount, pct_change, turnover
            FROM daily_bars
            WHERE symbol = ? AND trade_date <= ?
            ORDER BY trade_date DESC LIMIT ?
            """,
            conn, params=(stock_code, end_date, config["lookback_1y"]),
        )

        if len(df) < config["lookback_60d"]:
            return {}

        df = df.sort_values("trade_date").reset_index(drop=True)
        last60 = df.tail(config["lookback_60d"])

        # 60 日振幅
        high_60 = last60["high"].max()
        low_60  = last60["low"].min()
        avg_60  = last60["close"].mean()
        amplitude = (high_60 - low_60) / avg_60 if avg_60 > 0 else 0.0

        # 60 日均换手率（百分比形式）
        turnover_avg = float(last60["turnover"].mean()) if "turnover" in last60.columns else 0.0

        # 60 日均成交额
        amount_avg = float(last60["amount"].mean())

        # 近 1 年涨停次数（用板块感知阈值）
        name = _get_name_map(conn).get(stock_code, "")
        limit_pct = get_limit_pct(stock_code, name, end_date)
        # pct_change DB 里是百分比形式（如 6.30 = +6.30%）
        threshold = limit_pct - 0.1
        limit_up_count = int((df["pct_change"].fillna(0) >= threshold).sum())

        # 当日数据 + 量比
        today = df.iloc[-1]
        vol_window = df.tail(config["vol_ma_window"] + 1).iloc[:-1]["volume"]  # 用 today 之前的 N 日
        vol_ma = vol_window.mean() if len(vol_window) > 0 else 0
        today_vol_ratio = float(today["volume"]) / vol_ma if vol_ma > 0 else 0.0

        return {
            "symbol":             stock_code,
            "end_date":           end_date,
            "amplitude_60d":      round(float(amplitude), 4),
            "turnover_60d_avg":   round(turnover_avg, 4),
            "amount_60d_avg":     round(amount_avg, 0),
            "limit_up_count_1y":  limit_up_count,
            "today_volume_ratio": round(today_vol_ratio, 2),
            "today_turnover":     round(float(today.get("turnover", 0) or 0), 4),
            "today_pct_change":   round(float(today.get("pct_change", 0) or 0), 4),
            "data_complete":      turnover_avg > 0,  # turnover 数据是否完整
        }
    finally:
        if own_conn:
            conn.close()


# ════════════════════════════════════════
#  批量指标计算（性能版，回测主用）
# ════════════════════════════════════════

def calculate_zombie_metrics_batch(conn, symbols: list, end_date: str,
                                    config: dict) -> pd.DataFrame:
    """
    批量计算所有股票的僵尸股特征指标。

    Returns
    -------
    DataFrame，列：
        symbol, amplitude_60d, turnover_60d_avg, amount_60d_avg,
        limit_up_count_1y, today_volume_ratio, today_turnover,
        today_pct_change, data_complete
    """
    if not symbols:
        return pd.DataFrame()

    target_set = set(str(s).zfill(6) for s in symbols)
    name_map = _get_name_map(conn)

    # 一次查 240 天数据
    all_df = _load_window_data(conn, end_date, config["lookback_1y"])
    if all_df.empty:
        return pd.DataFrame()

    all_df["symbol"] = all_df["symbol"].astype(str).str.zfill(6)

    rows = []
    lookback_60d = config["lookback_60d"]
    vol_ma_window = config["vol_ma_window"]

    for sym, group in all_df.groupby("symbol", sort=False):
        if sym not in target_set:
            continue
        if len(group) < lookback_60d:
            continue

        group = group.sort_values("trade_date").reset_index(drop=True)
        last60 = group.tail(lookback_60d)

        # 60 日振幅
        high_60 = last60["high"].max()
        low_60  = last60["low"].min()
        avg_60  = last60["close"].mean()
        amplitude = (high_60 - low_60) / avg_60 if avg_60 > 0 else 0.0

        # 60 日均换手率
        turnover_avg = float(last60["turnover"].mean()) if "turnover" in last60.columns else 0.0

        # 60 日均成交额
        amount_avg = float(last60["amount"].mean())

        # 近 1 年涨停次数
        name = name_map.get(sym, "")
        limit_pct = get_limit_pct(sym, name, end_date)
        threshold = limit_pct - 0.1
        limit_up_count = int((group["pct_change"].fillna(0) >= threshold).sum())

        # 当日数据 + 量比
        today = group.iloc[-1]
        # 量比基准：今日之前的 N 日均量
        vol_window = group.tail(vol_ma_window + 1).iloc[:-1]["volume"]
        vol_ma = vol_window.mean() if len(vol_window) > 0 else 0
        today_vol_ratio = float(today["volume"]) / vol_ma if vol_ma > 0 else 0.0

        rows.append({
            "symbol":             sym,
            "amplitude_60d":      round(float(amplitude), 4),
            "turnover_60d_avg":   round(turnover_avg, 4),
            "amount_60d_avg":     round(amount_avg, 0),
            "limit_up_count_1y":  limit_up_count,
            "today_volume_ratio": round(today_vol_ratio, 2),
            "today_turnover":     round(float(today.get("turnover", 0) or 0), 4),
            "today_pct_change":   round(float(today.get("pct_change", 0) or 0), 4),
            "data_complete":      turnover_avg > 0,
        })

    return pd.DataFrame(rows)


# ════════════════════════════════════════
#  判定函数
# ════════════════════════════════════════

def is_zombie_stock(metrics: dict, config: dict) -> bool:
    """
    根据指标判定是否为僵尸股。
    返回 True = 是僵尸股，应过滤；False = 非僵尸，可参与策略扫描。

    判定逻辑：每个维度数据缺失时（turnover_60d_avg=0 或其它异常），
    根据 config['skip_missing_dimension'] 决定是否跳过该维度。
    """
    if not metrics:
        return False  # 无数据不强行判定（保守）

    skip_missing = config.get("skip_missing_dimension", True)
    votes = 0

    # 维度 1：60 日振幅
    amp = metrics.get("amplitude_60d", 0)
    if amp > 0 and amp < config["amplitude_60d_threshold"]:
        votes += 1

    # 维度 2：60 日均换手率
    turn = metrics.get("turnover_60d_avg", 0)
    if turn > 0:  # 有数据才计入
        if turn < config["turnover_60d_threshold"]:
            votes += 1
    elif not skip_missing:
        # 不跳过缺失 → 缺失视为"低换手"
        votes += 1

    # 维度 3：60 日均成交额
    amt = metrics.get("amount_60d_avg", 0)
    if amt > 0 and amt < config["amount_60d_threshold"]:
        votes += 1

    # 维度 4：近 1 年涨停次数
    lu = metrics.get("limit_up_count_1y", -1)
    if lu >= 0 and lu <= config["limit_up_1y_threshold"]:
        votes += 1

    return votes >= config["min_conditions_to_filter"]


def is_awakening(metrics: dict, config: dict) -> bool:
    """
    检测是否处于觉醒状态（即便是僵尸股也保留观察）。
    任一觉醒信号成立即返回 True。
    """
    if not metrics:
        return False

    vol_ratio = metrics.get("today_volume_ratio", 0)
    turnover  = metrics.get("today_turnover", 0)
    pct       = metrics.get("today_pct_change", 0)

    return (
        vol_ratio > config["awakening_volume_ratio"]
        or turnover > config["awakening_turnover"]
        or pct      > config["awakening_pct_change"]
    )


# ════════════════════════════════════════
#  主入口：批量过滤
# ════════════════════════════════════════

def filter_stock_pool(conn, symbols: list, end_date: str,
                      config: dict | None = None,
                      return_diagnostics: bool = False):
    """
    批量过滤股票池。

    Parameters
    ----------
    conn : sqlite3.Connection
    symbols : list[str]   待过滤的股票代码（6 位）
    end_date : str        基准交易日 'YYYY-MM-DD'
    config : dict         覆盖默认配置（None 时用 zombie_config.ZOMBIE_CONFIG）
    return_diagnostics : bool
        True 时返回 (kept_symbols, diagnostics_df)，False 时只返回 kept_symbols

    Returns
    -------
    list[str]            通过过滤的 symbols（保留扫描）
    or (list, DataFrame) 当 return_diagnostics=True
    """
    if config is None:
        from zombie_config import ZOMBIE_CONFIG
        config = ZOMBIE_CONFIG

    if not config.get("enabled", True):
        if return_diagnostics:
            return list(symbols), pd.DataFrame()
        return list(symbols)

    metrics_df = calculate_zombie_metrics_batch(conn, symbols, end_date, config)
    if metrics_df.empty:
        # 数据不足，全部保留（保守）
        if return_diagnostics:
            return list(symbols), pd.DataFrame()
        return list(symbols)

    # 逐行判定
    metrics_df["is_zombie"]    = metrics_df.apply(lambda r: is_zombie_stock(r.to_dict(), config), axis=1)
    metrics_df["is_awakening"] = metrics_df.apply(lambda r: is_awakening(r.to_dict(), config), axis=1)
    # 过滤规则：是僵尸 且 不在觉醒 → 剔除
    metrics_df["filtered"] = metrics_df["is_zombie"] & ~metrics_df["is_awakening"]

    kept_symbols = metrics_df.loc[~metrics_df["filtered"], "symbol"].tolist()

    # 数据不足无法判定的（不在 metrics_df 里）保守保留
    not_in_df = set(str(s).zfill(6) for s in symbols) - set(metrics_df["symbol"])
    kept_symbols.extend(sorted(not_in_df))

    n_zombie = int(metrics_df["is_zombie"].sum())
    n_filtered = int(metrics_df["filtered"].sum())
    n_awakening = int(metrics_df["is_awakening"].sum())
    logger.info(
        "[zombie_filter] %s: 输入 %d 只，僵尸 %d 只，觉醒 %d 只，最终过滤 %d 只，保留 %d 只",
        end_date, len(symbols), n_zombie, n_awakening, n_filtered, len(kept_symbols),
    )

    if return_diagnostics:
        return kept_symbols, metrics_df
    return kept_symbols
