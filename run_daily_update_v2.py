"""
run_daily_update_v2.py
日常盘后数据更新（Tushare + BaoStock 双通道）

用法:
    python run_daily_update_v2.py              # 自动找最近交易日更新
    python run_daily_update_v2.py 2026-04-01   # 更新指定日期
"""

import sys
import os
import logging
import pandas as pd
from datetime import date, datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from data.fetcher_daily import fetch_daily_update, fetch_all_via_tushare
from data.storage import save_daily_bars_bulk, get_connection, init_database
from data.trade_calendar import (
    ensure_calendar_loaded,
    latest_trading_day_on_or_before,
    next_trading_day,
)

# ========== 日志配置 ==========
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            f"logs/daily_update_{date.today().strftime('%Y%m%d')}.log",
            encoding="utf-8",
        ),
    ],
)
logger = logging.getLogger(__name__)

# 清理 30 天前的日志（只动 daily_update_ / signal_ / missing_ 开头的，回测产物保留）
_removed = config.cleanup_old_logs()
if _removed:
    logger.info("已清理 %d 个 30 天前的旧日志", _removed)

# 确保 trade_calendar 表存在（如果是老 DB 需要补建表），并自动拉取一次日历
init_database()  # IF NOT EXISTS，已存在不会重建
ensure_calendar_loaded(getattr(config, "TUSHARE_TOKEN", ""))


def _maybe_refresh_stock_list():
    """每周一自动刷新一次股票名（避免 ST 戴帽/摘帽信息过时）"""
    import sqlite3
    if date.today().weekday() != 0:  # 0 = 周一
        return
    try:
        conn = get_connection()
        row = conn.execute(
            "SELECT MAX(updated_at) FROM stock_info"
        ).fetchone()
        conn.close()
        last_update = row[0] if row and row[0] else ""
        # 已经今天更新过就不再跑
        today_str = date.today().strftime("%Y-%m-%d")
        if last_update.startswith(today_str):
            return
    except Exception:
        pass

    try:
        from data.universe import update_stock_list
        logger.info("[周一] 自动刷新股票名（ST 状态等）")
        n = update_stock_list()
        if n > 0:
            logger.info("[周一] 股票池已刷新：%d 只", n)
        else:
            logger.warning("[周一] 股票池刷新失败，使用旧名字（下周再试）")
    except Exception as e:
        logger.warning("[周一] 股票池刷新异常：%s", e)


_maybe_refresh_stock_list()


def _get_min_complete_count(conn=None) -> int:
    """
    动态算"完整数据"阈值：股票池总数 × 90%。
    比固定 3000 更稳健——避免 Tushare 部分故障返回 4000 条仍被错认为"已完整"。
    """
    own_conn = conn is None
    if own_conn:
        conn = get_connection()
    try:
        total = conn.execute("SELECT COUNT(*) FROM stock_info").fetchone()[0]
        if total and total > 0:
            return max(int(total * 0.9), 100)
    except Exception:
        pass
    finally:
        if own_conn:
            conn.close()
    return 3000  # 兜底


# 启动时算一次，存到模块级
MIN_COMPLETE_COUNT = _get_min_complete_count()
logger.info("[数据完整阈值] %d 条（基于股票池 90%%）", MIN_COMPLETE_COUNT)


def get_date_record_count(trade_date: str) -> int:
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT COUNT(*) FROM daily_bars WHERE trade_date = ?",
            (trade_date,),
        )
        return cursor.fetchone()[0]
    except Exception:
        return 0
    finally:
        conn.close()


def get_latest_db_date() -> str | None:
    """从数据库找最近一个数据完整的交易日（按动态 MIN_COMPLETE_COUNT）"""
    conn = get_connection()
    try:
        min_cnt = _get_min_complete_count(conn)
        df = pd.read_sql("""
            SELECT trade_date, COUNT(*) as cnt
            FROM daily_bars
            GROUP BY trade_date
            HAVING cnt >= ?
            ORDER BY trade_date DESC
            LIMIT 1
        """, conn, params=(min_cnt,))
        if not df.empty:
            return df.iloc[0]["trade_date"]
        return None
    except Exception:
        return None
    finally:
        conn.close()


def _next_workday_after(d_str: str) -> str:
    """跳到 d_str 之后的下一个工作日（仅按周一-周五，不识别假期）"""
    d = date.fromisoformat(d_str) + timedelta(days=1)
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d.strftime("%Y-%m-%d")


def find_target_date(specified: str | None,
                     skip_dates: set | None = None) -> str | None:
    """
    确定本次要更新的目标日期。
    优先用 trade_calendar 表识别真正的交易日（跳过法定节假日）；
    日历为空时回落到旧逻辑（仅按周一-周五判断）。

    skip_dates: 本次运行中已确认空数据的日期（运行时累积，不持久化）
    """
    if specified:
        return specified

    skip_dates = skip_dates or set()
    today_str = date.today().strftime("%Y-%m-%d")

    # 用日历找今天或之前最近的开市日
    candidate_str = latest_trading_day_on_or_before(today_str)

    if candidate_str is None:
        # 日历未填充，回落到旧逻辑（按周一-周五）
        candidate = date.today()
        for _ in range(7):
            if candidate.weekday() < 5:
                break
            candidate -= timedelta(days=1)
        candidate_str = candidate.strftime("%Y-%m-%d")

    # 候选日已知是假期 → 再往前找
    while candidate_str in skip_dates:
        cd = date.fromisoformat(candidate_str) - timedelta(days=1)
        while cd.weekday() >= 5:
            cd -= timedelta(days=1)
        candidate_str = cd.strftime("%Y-%m-%d")

    latest_db = get_latest_db_date()
    logger.info(f"数据库最新完整交易日：{latest_db}")
    logger.info(f"目标更新日期：{candidate_str}")

    if latest_db and latest_db >= candidate_str:
        logger.info("数据库已是最新，无需更新")
        return None

    if latest_db:
        next_str = next_trading_day(latest_db) or _next_workday_after(latest_db)

        # 跳过运行时已知的假期日（最多前进 30 天，防失控）
        guard = 30
        while next_str in skip_dates and guard > 0:
            next_str = _next_workday_after(next_str)
            cal_hint = next_trading_day(next_str)
            if cal_hint and cal_hint > next_str:
                next_str = cal_hint
            guard -= 1

        if next_str <= candidate_str:
            logger.info(f"数据库落后，从 {next_str} 开始补（今日目标 {candidate_str}）")
            return next_str

    return candidate_str


def get_stock_list_from_db() -> list:
    conn = get_connection()
    try:
        df = pd.read_sql_query("SELECT symbol FROM stock_info", conn)
        return df["symbol"].astype(str).str.zfill(6).tolist()
    except Exception as e:
        logger.error(f"从数据库获取股票列表失败: {e}")
        return []
    finally:
        conn.close()


def update_one_date(trade_date: str):
    """更新单个交易日的数据"""
    logger.info("=" * 60)
    logger.info(f"开始日常更新: {trade_date}")
    logger.info(f"数据源策略: {config.DAILY_UPDATE_SOURCE}")
    logger.info("=" * 60)

    existing_count = get_date_record_count(trade_date)
    if existing_count >= MIN_COMPLETE_COUNT:
        logger.info(f"数据库已有 {trade_date} 的数据 ({existing_count} 条)，无需更新")
        return True

    if existing_count > 0:
        logger.warning(f"数据库 {trade_date} 仅有 {existing_count} 条（不足 {MIN_COMPLETE_COUNT}），需要补全")

    try:
        from data.universe import get_stock_list
        stock_codes = get_stock_list()
    except Exception:
        stock_codes = get_stock_list_from_db()

    if not stock_codes:
        logger.warning("股票池为空，尝试直接拉取全市场")

    logger.info(f"股票池: {len(stock_codes)} 只")

    tushare_token = getattr(config, "TUSHARE_TOKEN", "")
    source = getattr(config, "DAILY_UPDATE_SOURCE", "auto")

    if stock_codes:
        df = fetch_daily_update(
            stock_codes=stock_codes,
            trade_date=trade_date,
            source=source,
            tushare_token=tushare_token,
        )
    else:
        if tushare_token:
            df = fetch_all_via_tushare(tushare_token, trade_date)
        else:
            logger.error("无股票池且无 Tushare Token，无法更新")
            return False

    if df.empty:
        # 区分"数据尚未发布"和"真•非交易日"：
        #   工作日 + 当前时间早于 17:30 → 大概率是数据没出来
        #   工作日但已晚 → 大概率法定节假日
        #   周末 → 周末非交易日
        try:
            target = date.fromisoformat(trade_date)
        except ValueError:
            target = None

        now = datetime.now()
        is_weekday = target is not None and target.weekday() < 5
        is_today   = target == date.today()
        too_early  = now.hour < 17 or (now.hour == 17 and now.minute < 30)

        logger.warning("=" * 60)
        if is_weekday and is_today and too_early:
            logger.warning("⚠️  %s 暂无数据 — 今日盘后数据尚未发布", trade_date)
            logger.warning("    Tushare 通常 17:00 后稳定，BaoStock 通常 17:30 后稳定")
            logger.warning("    建议：等到 17:30 之后再跑 python run_daily_update_v2.py")
        elif is_weekday:
            logger.warning("⚠️  %s 工作日但接口无数据 — 可能是法定节假日", trade_date)
            logger.warning("    或数据接口异常 / Tushare 积分不足")
        else:
            logger.warning("%s 周末非交易日，跳过", trade_date)
        logger.warning("=" * 60)
        return False

    logger.info(f"共获取 {len(df)} 条记录，覆盖 {df['code'].nunique()} 只股票")

    saved = save_daily_bars_bulk(df)
    logger.info(f"写入完成: {saved} 条新增")

    final_count = get_date_record_count(trade_date)
    logger.info(f"验证: {trade_date} 现有 {final_count} 条记录")

    if stock_codes:
        fetched_codes = set(df["code"].astype(str).str.zfill(6))
        all_codes = set(c.zfill(6) for c in stock_codes)
        missing = all_codes - fetched_codes
        if missing:
            logger.warning(f"缺失 {len(missing)} 只股票（停牌/退市/异常）")
            missing_file = f"logs/missing_{trade_date.replace('-', '')}.txt"
            with open(missing_file, "w") as f:
                f.write("\n".join(sorted(missing)))
            logger.info(f"缺失列表已保存: {missing_file}")

    logger.info(f"{trade_date} 更新完成 ✅")
    return True


MAX_AUTO_LOOP = 30   # 自动补 gap 的最大轮次（防失控）


def main():
    specified = sys.argv[1] if len(sys.argv) > 1 else None

    # 指定日期 → 只跑这一天，不循环
    if specified:
        update_one_date(specified)
        return

    # 自动模式：循环补到追平为止
    # skip_dates 累积本次运行已确认无数据的日期（假期或数据未发布）
    skip_dates: set[str] = set()

    for round_idx in range(1, MAX_AUTO_LOOP + 1):
        target_date = find_target_date(None, skip_dates)
        if target_date is None:
            return  # find_target_date 内部已打印"已是最新"

        # 兜底：如果 find_target_date 居然又返回了已知 skip 的日期（不应该发生），强制结束
        if target_date in skip_dates:
            logger.warning("⚠️ 目标 %s 已在 skip 集，find_target_date 逻辑有问题，停止", target_date)
            return

        logger.info("───── 自动补 gap 第 %d 轮：%s ─────", round_idx, target_date)
        success = update_one_date(target_date)

        if not success:
            # 这一天确认无数据（假期/未发布）→ 加入 skip 集，下轮 find_target_date 会跳过它
            skip_dates.add(target_date)
            logger.info("将 %s 加入跳过集，继续下一轮", target_date)
            continue
    else:
        logger.warning("达到最大轮次 %d，仍未追平。请检查数据接口", MAX_AUTO_LOOP)


if __name__ == "__main__":
    main()
