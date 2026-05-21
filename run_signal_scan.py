"""
每日信号扫描入口
运行方式：python run_signal_scan.py

流程：
  1. 先增量更新当日行情数据
  2. 扫描全市场信号
  3. 推送微信

建议配置 Windows 定时任务：每天 15:30 自动运行
"""

import sys
import os
import logging
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from config import LOG_DIR

log_file = os.path.join(LOG_DIR, f"signal_{datetime.today().strftime('%Y%m%d')}.log")
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

# 清理 30 天前的日志
_removed = config.cleanup_old_logs()
if _removed:
    logger.info("已清理 %d 个 30 天前的旧日志", _removed)


def run():
    logger.info("═" * 50)
    logger.info("每日信号扫描开始：%s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("═" * 50)

    # ── Step 1：增量更新当日数据 ──
   # logger.info("Step 1：更新当日行情数据...")
    #try:
        #from data.storage import init_database, get_data_summary
       # from data.universe import get_all_symbols
        #from data.fetcher import batch_update_universe

      #  init_database()
       # symbols = get_all_symbols()
      #  if not symbols:
      #      logger.error("股票列表为空，请先运行 run_init_database.py")
       #     return

      #  stats = batch_update_universe(
       #     symbols=symbols,
       #     incremental=True,
      #      sleep_interval=0.2,
      #      resume=False,   # 增量更新不需要断点续传
        #)
      #  logger.info("数据更新完成：新增 %d 条记录", #stats.get("new_records", 0))
   # except Exception as e:
       # logger.error("数据更新失败：%s，继续用昨日数据扫描", e)

    # ── Step 2：扫描信号 ──
    logger.info("Step 2：扫描全市场信号...")
    from signals.scanner import run_scan, push_to_wechat, STRATEGIES

    scan_results = run_scan(top_n=20)

    # 统计
    total = sum(len(v) for v in scan_results.values())
    logger.info("扫描完成，共触发 %d 只信号", total)

    # ── Step 3：推送微信 ──
    logger.info("Step 3：推送微信...")
    push_to_wechat(scan_results, STRATEGIES)

    logger.info("═" * 50)
    logger.info("每日信号扫描完成")
    logger.info("═" * 50)


if __name__ == "__main__":
    run()
