"""
zombie_config.py — 僵尸股过滤层配置

判定逻辑：4 个维度，满足 N 个则判定为"僵尸股"被过滤。
觉醒识别：即便满足僵尸条件，当日触发觉醒信号则不过滤（标记为高风险机会）。

阈值用法（v1 仅 4 维，跳过股性评分 + 自由流通市值）：
"""

from __future__ import annotations

ZOMBIE_CONFIG: dict = {
    # ── 总开关 ──
    "enabled": True,

    # ── 4 个判定指标的阈值（"小于"该值算"僵尸"特征）──
    "amplitude_60d_threshold": 0.30,    # 60 日振幅 < 30%
    "turnover_60d_threshold":  1.00,    # 60 日均换手率 < 1%（百分比形式，与 DB 一致）
    "amount_60d_threshold":    1e8,     # 60 日均成交额 < 1 亿元
    "limit_up_1y_threshold":   0,       # 近 1 年涨停次数 = 0

    # ── 满足几个条件才算僵尸股 ──
    "min_conditions_to_filter": 2,

    # ── 觉醒识别（即使是僵尸股，触发以下任一不过滤）──
    "awakening_volume_ratio":  5.0,     # 量比 > 5（今日量 / 20 日均量）
    "awakening_turnover":      5.0,     # 实换手 > 5%（百分比形式）
    "awakening_pct_change":    5.0,     # 涨幅 > 5%（百分比形式）

    # ── 计算窗口 ──
    "lookback_60d": 60,                 # 60 个交易日
    "lookback_1y":  240,                # ~1 年 ≈ 240 交易日
    "vol_ma_window": 20,                # 量比基准窗口

    # ── 数据缺失容错 ──
    # turnover 字段在某些时段缺失（=0），开启时缺失维度自动跳过、不计入 vote
    "skip_missing_dimension": True,
}
