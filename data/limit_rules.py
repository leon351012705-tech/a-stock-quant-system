"""
data/limit_rules.py — A股涨跌停规则查询

按板块和ST状态返回单日涨跌幅上限（百分比，正数）。

板块判定依据股票代码前缀：
  60xxxx          : 沪市主板
  000/001/002/003 : 深市主板（中小板已并入）
  300/301         : 创业板
  688/689         : 科创板
  4xx / 8xx       : 北交所

涨跌停规则：
  主板（沪深）        : ±10%
  创业板 300/301      : ±20% (2020-08-24 起；之前为 ±10%)
  科创板 688/689      : ±20%（2019-07-22 开市起即为 20%）
  北交所              : ±30%
  ST/*ST              : ±5%（不分板块，ST 前缀优先）

使用：
  from data.limit_rules import get_limit_pct, is_limit_move
  pct = get_limit_pct("300750", "宁德时代", "2026-04-28")  # → 20.0
  pct = get_limit_pct("600519", "贵州茅台")                # → 10.0
  pct = get_limit_pct("600519", "*ST 茅台")                # → 5.0
"""

from __future__ import annotations

# 创业板 20% 涨跌幅生效日（A 股注册制改革）
CHINEXT_20PCT_DATE = "2020-08-24"

# 涨跌停判定容差（小数点后波动）：当 |pct| >= limit - tolerance 视为涨跌停
LIMIT_TOLERANCE = 0.1


def is_st(name: str) -> bool:
    """名称包含 ST / *ST / S*ST 等都算 ST 股。"""
    if not name:
        return False
    s = str(name).upper().replace(" ", "")
    return ("ST" in s)


def get_limit_pct(symbol: str, name: str = "", trade_date: str | None = None) -> float:
    """
    返回某股某日的涨跌停百分比（正数）。

    参数：
        symbol     : 6 位股票代码，如 '600519' / '300750' / '688008'
        name       : 股票名称（用于识别 ST），可为空
        trade_date : 'YYYY-MM-DD'，用于创业板历史规则。None=按当前规则

    注意：
        ST 状态来自传入的 name，调用方应传入"该交易日的名字"。
        当前 stock_info 表只存最新名字，深度历史回测会有边界误差，
        但 2020 年后的数据通常无大碍。
    """
    # ST 优先（覆盖所有板块）
    if is_st(name):
        return 5.0

    s = str(symbol).strip().zfill(6)

    # 北交所
    if s.startswith(("4", "8")):
        return 30.0

    # 科创板（一直是 20%）
    if s.startswith(("688", "689")):
        return 20.0

    # 创业板（2020-08-24 起 20%）
    if s.startswith(("300", "301")):
        if trade_date and trade_date < CHINEXT_20PCT_DATE:
            return 10.0
        return 20.0

    # 主板沪深
    return 10.0


def is_limit_move(pct_change: float, limit_pct: float,
                  tolerance: float = LIMIT_TOLERANCE) -> bool:
    """
    判定是否涨跌停。
    用 |pct| >= limit - tolerance 而不是 == limit，是因为：
      - 数据源精度可能让 +10.00% 显示为 +9.98% 等
      - tolerance=0.1 给一点缓冲，避免漏判
    """
    try:
        return abs(float(pct_change)) >= (limit_pct - tolerance)
    except (TypeError, ValueError):
        return False
