"""
【数据层】fetcher.py — akshare 数据拉取模块（多线程并发提速版）

职责：
  - 拉取单只/全市场股票日线数据
  - 支持增量更新
  - 支持断点续传
  - 支持重试、随机等待、熔断保护
  - 以“最新交易日”作为更新目标
  - 使用线程池并发提速（不改变数据接口，不影响回测）

说明：
  - 仍使用 stock_zh_a_hist 拉取前复权数据
  - 只是并发执行，多线程不会改变数据质量
"""

import akshare as ak
import pandas as pd
import logging
import sys
import os
import json
import time
import random
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

# 禁用系统代理，避免 akshare 访问东方财富时走代理导致失败
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)

# 保持路径引用正确
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DEFAULT_START_DATE, DEFAULT_ADJUST, LOG_DIR
from data.storage import save_daily_bars, get_latest_date

logger = logging.getLogger(__name__)

# 断点续传进度文件路径
PROGRESS_FILE = os.path.join(LOG_DIR, "batch_fetch_progress.json")

# 并发线程数：建议 8~16，先保守一点
MAX_WORKERS = 3


# ════════════════════════════════════════
#  工具函数
# ════════════════════════════════════════

def _to_db_date(date_str: str) -> str:
    """将日期统一转成 YYYY-MM-DD。"""
    return pd.to_datetime(date_str).strftime("%Y-%m-%d")


def _to_api_date(date_str: str) -> str:
    """将日期统一转成 YYYYMMDD。"""
    return pd.to_datetime(date_str).strftime("%Y%m%d")


def get_market_latest_trade_date() -> str:
    """
    获取市场最新交易日（YYYY-MM-DD）。

    当前实现：
      使用上证指数日线作为市场交易日参考。
    """
    try:
        df = ak.stock_zh_index_daily_em(symbol="sh000001")
        if df is None or df.empty:
            raise ValueError("上证指数数据为空")

        if "date" in df.columns:
            latest = df["date"].iloc[-1]
        elif "日期" in df.columns:
            latest = df["日期"].iloc[-1]
        else:
            raise ValueError(f"无法识别指数日期列，当前列名: {list(df.columns)}")

        latest = _to_db_date(latest)
        logger.info("市场最新交易日：%s", latest)
        return latest

    except Exception as e:
        logger.error("获取市场最新交易日失败：%s", e)
        fallback = (datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d")
        logger.warning("回退使用兜底日期：%s", fallback)
        return fallback


# ════════════════════════════════════════
#  单只股票拉取
# ════════════════════════════════════════

def fetch_and_save_stock(
    symbol: str,
    target_trade_date: str,
    start_date: str = None,
    adjust: str = DEFAULT_ADJUST,
    incremental: bool = True,
) -> int:
    """
    拉取单只股票数据并写入数据库。

    返回：
        >=0 : 实际新增记录数
        -1  : 最终失败
    """
    db_latest = get_latest_date(symbol)

    if incremental:
        if db_latest:
            if db_latest >= target_trade_date:
                return 0

            next_day = datetime.strptime(db_latest, "%Y-%m-%d") + timedelta(days=1)
            start_date = next_day.strftime("%Y%m%d")
        else:
            start_date = start_date or DEFAULT_START_DATE
    else:
        start_date = start_date or DEFAULT_START_DATE

    end_date = _to_api_date(target_trade_date)

    try:
        if datetime.strptime(start_date, "%Y%m%d") > datetime.strptime(end_date, "%Y%m%d"):
            return 0
    except Exception as e:
        logger.error("[%s] 日期格式异常：start=%s, end=%s, error=%s", symbol, start_date, end_date, e)
        return -1

    df = None
    max_retries = 3

    for attempt in range(max_retries):
        try:
            # 重试时加小抖动
            if attempt > 0:
                wait_time = (attempt * 3) + random.uniform(1, 4)
                logger.warning("[%s] 连接异常，等待 %.1f 秒后进行第 %d 次重试...", symbol, wait_time, attempt + 1)
                time.sleep(wait_time)

            df = ak.stock_zh_a_hist(
                symbol=symbol,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust=adjust,
            )
            break

        except Exception as e:
            if attempt == max_retries - 1:
                logger.error("[%s] 最终拉取失败（尝试 %d 次）: %s", symbol, max_retries, e)
                return -1

    if df is None or df.empty:
        return 0

    inserted = save_daily_bars(df, symbol)
    return inserted


def _fetch_worker(args: tuple) -> tuple:
    """
    线程池 worker。

    返回：
        (symbol, count)
    """
    symbol, target_trade_date, incremental, adjust = args
    count = fetch_and_save_stock(
        symbol=symbol,
        target_trade_date=target_trade_date,
        incremental=incremental,
        adjust=adjust,
    )
    return symbol, count


# ════════════════════════════════════════
#  断点续传
# ════════════════════════════════════════

def _load_progress() -> dict:
    """读取断点续传进度文件。"""
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning("读取进度文件失败，已忽略：%s", e)
    return {}


def _save_progress(progress: dict):
    """保存断点续传进度文件。"""
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)


# ════════════════════════════════════════
#  批量拉取（并发版）
# ════════════════════════════════════════

def batch_update_universe(
    symbols: list = None,
    incremental: bool = True,
    sleep_interval: float = 0.6,
    resume: bool = True,
    max_workers: int = MAX_WORKERS,
    target_trade_date: str = None,
) -> dict:
    """
    批量更新全市场股票数据（并发版）。

    参数：
        symbols         : 股票代码列表
        incremental     : True=增量更新
        sleep_interval  : 预留参数（并发版中仅用于批次间小休息）
        resume          : True=启用断点续传
        max_workers     : 线程数

    返回：
        stats 字典
    """
    if symbols is None:
        from data.universe import get_all_symbols
        symbols = get_all_symbols()
        if not symbols:
            logger.error("股票列表为空")
            return {}

    if target_trade_date is None:
        target_trade_date = get_market_latest_trade_date()

    total = len(symbols)
    progress = _load_progress() if resume else {}
    done_set = set(progress.get("done", []))
    failed_set = set(progress.get("failed", []))

    remaining = [s for s in symbols if s not in done_set]

    logger.info(
        "🚀 启动拉取任务 | 总数: %d | 已完成: %d | 待处理: %d | 断点续传: %s | 目标交易日: %s | 并发线程: %d",
        total, len(done_set), len(remaining), resume, target_trade_date, max_workers
    )

    stats = {
        "success": 0,
        "skipped": 0,
        "failed": 0,
        "new_records": 0,
        "done": 0,
        "target_trade_date": target_trade_date,
    }

    if not remaining:
        logger.info("当前无待处理股票")
        return stats

    # 分批提交，避免一次扔 5000 个 future
    batch_size = max_workers * 20

    for batch_start in range(0, len(remaining), batch_size):
        batch_symbols = remaining[batch_start: batch_start + batch_size]

        logger.info(
            "处理批次 %d ~ %d / %d",
            batch_start + 1,
            min(batch_start + len(batch_symbols), len(remaining)),
            len(remaining)
        )

        tasks = [
            (symbol, target_trade_date, incremental, DEFAULT_ADJUST)
            for symbol in batch_symbols
        ]

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {executor.submit(_fetch_worker, task): task[0] for task in tasks}

            for idx, future in enumerate(as_completed(future_map), 1):
                symbol = future_map[future]

                try:
                    symbol, count = future.result()

                    if count == -1:
                        stats["failed"] += 1
                        failed_set.add(symbol)
                    else:
                        if count > 0:
                            stats["success"] += 1
                            stats["new_records"] += count
                        else:
                            stats["skipped"] += 1

                        done_set.add(symbol)
                        stats["done"] += 1
                        if symbol in failed_set:
                            failed_set.remove(symbol)

                except Exception as e:
                    stats["failed"] += 1
                    failed_set.add(symbol)
                    logger.error("[%s] 并发任务异常：%s", symbol, e)

                # 每完成一部分打印一次进度
                processed_now = batch_start + idx
                if processed_now % 200 == 0 or processed_now == len(remaining):
                    pct = processed_now / len(remaining) * 100
                    logger.info(
                        "批量进度: %.1f%% | 成功:%d 跳过:%d 失败:%d 已完成:%d",
                        pct, stats["success"], stats["skipped"], stats["failed"], stats["done"]
                    )

        # 每个批次结束保存一次进度
        _save_progress({
            "done": list(done_set),
            "failed": list(failed_set),
            "target_trade_date": target_trade_date,
        })

        # 批次间短暂休息，避免压太狠
        time.sleep(sleep_interval)

    logger.info(
        "✅ 批量更新完成 | 目标交易日:%s | 新增成功:%d | 跳过:%d | 失败:%d | 新增记录:%d | 本轮完成:%d/%d",
        stats["target_trade_date"],
        stats["success"],
        stats["skipped"],
        stats["failed"],
        stats["new_records"],
        stats["done"],
        total
    )

    return stats


def retry_failed(sleep_interval: float = 1.0, max_workers: int = MAX_WORKERS) -> dict:
    """
    重试上次失败的股票。
    """
    progress = _load_progress()
    failed_list = progress.get("failed", [])

    if not failed_list:
        logger.info("没有失败记录需要重试")
        return {}

    logger.info("开始重试失败股票，共 %d 只", len(failed_list))
    return batch_update_universe(
        symbols=failed_list,
        incremental=True,
        sleep_interval=sleep_interval,
        resume=False,
        max_workers=max_workers,
    )