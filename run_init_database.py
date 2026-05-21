"""
首次全量建库脚本（只需运行一次）
运行方式：python run_init_database.py

功能：
  1. 初始化数据库表结构
  2. 拉取全市场股票列表
  3. 对全市场所有股票拉取历史数据（从 config.DEFAULT_START_DATE 开始）
  4. 支持断点续传，中途关闭重新运行会接着上次继续

注意：
  - 全市场约 5000+ 只股票，首次全量建库耗时较长
  - 建议在网络稳定时运行
  - 中途按 Ctrl+C 可以随时中断，下次运行自动续接
  - 若要“完全重头建库”，请先删除 logs/batch_fetch_progress.json
"""

import sys
import os
import logging
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import LOG_DIR

log_file = os.path.join(
    LOG_DIR,
    f"init_database_{datetime.today().strftime('%Y%m%d_%H%M%S')}.log"
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file, encoding="utf-8"),
    ]
)
logger = logging.getLogger(__name__)


def run():
    from data.storage import init_database, get_data_summary
    from data.universe import update_stock_list, get_all_symbols, get_universe_summary
    from data.fetcher import batch_update_universe

    logger.info("═" * 50)
    logger.info("首次全量建库开始")
    logger.info("═" * 50)

    # ── Step 1：初始化数据库 ──
    logger.info("Step 1：初始化数据库...")
    init_database()

    # ── Step 2：拉取全市场股票列表 ──
    logger.info("Step 2：拉取全市场股票列表...")
    update_stock_list()
    summary = get_universe_summary()
    logger.info("股票列表更新完成：%s", summary)

    # ── Step 3：全量拉取历史数据 ──
    symbols = get_all_symbols()
    if not symbols:
        logger.error("股票列表为空，无法执行全量建库")
        return

    logger.info("Step 3：开始全量拉取 %d 只股票（支持断点续传）...", len(symbols))
    logger.info("提示：可随时按 Ctrl+C 中断，下次运行会从断点继续")
    logger.info("提示：若要完全重头建库，请先删除 logs/batch_fetch_progress.json")

    stats = batch_update_universe(
        symbols=symbols,
        incremental=False,   # 全量：从 DEFAULT_START_DATE 开始拉
        sleep_interval=1.0,  # 建库时稳定优先
        resume=True,         # 开启断点续传
    )

    # ── Step 4：完成摘要 ──
    logger.info("─" * 50)
    logger.info("建库完成摘要：")
    logger.info("  新增成功：%d 只", stats.get("success", 0))
    logger.info("  本次无新增：%d 只", stats.get("skipped", 0))
    logger.info("  失败：%d 只", stats.get("failed", 0))
    logger.info("  新增记录：%d 条", stats.get("new_records", 0))
    logger.info("  本轮完成：%d 只", stats.get("done", 0))
    logger.info("  目标交易日：%s", stats.get("target_trade_date", "未知"))

    db_summary = get_data_summary()
    if not db_summary.empty:
        logger.info("数据库覆盖 %d 只股票", len(db_summary))

    logger.info("═" * 50)
    logger.info("建库完成！后续每天只需运行 run_daily_update.py 做增量更新")
    logger.info("═" * 50)


if __name__ == "__main__":
    run()