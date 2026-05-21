"""
check_cache.py - 查看signal_cache表内容
用法:
  python check_cache.py                        # 所有日期汇总
  python check_cache.py 2026-04-24             # 指定日期汇总
  python check_cache.py 2026-04-24 detail      # 指定日期+具体股票
  python check_cache.py 2026-04-24 detail macd # 指定日期+指定策略的股票
"""
import sys
import sqlite3
import pandas as pd
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DB_PATH

date       = sys.argv[1] if len(sys.argv) > 1 else None
mode       = sys.argv[2] if len(sys.argv) > 2 else None
filter_sid = sys.argv[3] if len(sys.argv) > 3 else None

conn = sqlite3.connect(DB_PATH)

if not date:
    # 所有日期汇总
    df = pd.read_sql("""
        SELECT trade_date, strategy_id, COUNT(*) as 命中数
        FROM signal_cache
        GROUP BY trade_date, strategy_id
        ORDER BY trade_date DESC
    """, conn)
    print("\n全部缓存汇总：")
    print(df.to_string(index=False))

elif mode != "detail":
    # 指定日期汇总
    df = pd.read_sql("""
        SELECT strategy_id, COUNT(*) as 命中数
        FROM signal_cache
        WHERE trade_date = ?
        GROUP BY strategy_id
    """, conn, params=(date,))
    print(f"\n{date} 缓存内容：")
    print(df.to_string(index=False))

else:
    # 指定日期的具体股票
    where = "WHERE c.trade_date = ?"
    params = [date]
    if filter_sid:
        where += " AND c.strategy_id = ?"
        params.append(filter_sid)

    df = pd.read_sql(f"""
        SELECT c.trade_date, c.strategy_id, c.symbol,
               COALESCE(s.name, c.symbol) as 股票名称,
               c.close as 收盘价,
               c.pct_change as 涨跌幅,
               c.amount_w as 成交额万
        FROM signal_cache c
        LEFT JOIN stock_info s ON c.symbol = s.symbol
        {where}
        ORDER BY c.strategy_id, c.amount_w DESC
    """, conn, params=params)

    if df.empty:
        print(f"\n{date} 无缓存数据")
    else:
        print(f"\n{date} 详细命中记录（共{len(df)}条）：")
        print(df.to_string(index=False))

conn.close()
