"""
【回测层】engine.py — 单标的事件驱动回测引擎

设计原则：
  - 严格遵守 A 股规则：T+1、涨跌停无法成交、前复权数据
  - 手续费：买入佣金 + 卖出佣金 + 卖出印花税（含最低5元佣金）
  - 成交价格：使用次日开盘价（更接近实盘）
  - 仓位管理：每次买入使用 95% 可用资金
  - 风险控制：统一硬止损（默认 5%）

进阶功能（可选，不启用时完全兼容旧策略）：
  - 固定止盈
  - 跟踪止盈（冲高后回撤出场）
  - 动态时间评估（有肉继续拿，没肉走）
  - 绝对持仓上限

⚠️ 跟 run_resonance_backtest.py 的 simulate_trade() 区别：
  - 本引擎止损用 close 价（row["close"] <= stop_price）—— **保守**
  - run_resonance_backtest 用 low 价（row["low"] <= stop_price）—— **激进**
  两种各有道理：close 假设你不盯盘只看收盘；low 假设你设了真实止损单。
  实盘选哪个看你是否盯盘——A 股 T+1 卖出限制下，盘中触发 stop 的实际能不能立刻成交也是问题。
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

# ── 手续费参数（与主流券商对齐）──
BUY_COMMISSION  = 0.000086   # 买入佣金：万0.86
SELL_COMMISSION = 0.000086   # 卖出佣金：万0.86
STAMP_DUTY      = 0.001      # 印花税 0.1%（仅卖出收取）
MIN_COMMISSION  = 5.0        # 最低佣金 5 元/笔


class Backtester:
    """
    单标的向量化回测引擎。

    基础参数（所有策略通用）：
        df              : 日线数据 DataFrame
        initial_cash    : 初始资金，默认 10 万
        symbol          : 股票代码，仅用于日志
        position_pct    : 每次买入占可用资金比例，默认 0.95
        stop_loss_pct   : 硬止损比例，默认 0.05（5%）

    进阶参数（默认 None = 不启用，旧策略完全不受影响）：
        take_profit_pct        : 固定止盈，如 0.04（+4% 出场）
        trailing_trigger_pct   : 跟踪止盈激活阈值，如 0.06（+6% 激活）
        trailing_stop_pct      : 跟踪止盈回撤，如 0.02（从最高回落2%出场）
        max_hold_days          : 绝对最大持仓天数，如 5
        dynamic_exit_start_day : 动态评估起始天数，如 3（第3天开始评估）
    """

    def __init__(
        self,
        df: pd.DataFrame,
        initial_cash: float = 100000.0,
        symbol: str = "",
        position_pct: float = 0.95,
        stop_loss_pct: float = 0.05,
        take_profit_pct: float = None,
        trailing_trigger_pct: float = None,
        trailing_stop_pct: float = None,
        max_hold_days: int = None,
        dynamic_exit_start_day: int = None,
        limit_pct: float = None,
        name: str = "",
    ):
        self.df = df.copy().reset_index(drop=True)
        self.initial_cash = initial_cash
        self.symbol = symbol
        self.position_pct = position_pct
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.trailing_trigger_pct = trailing_trigger_pct
        self.trailing_stop_pct = trailing_stop_pct
        self.max_hold_days = max_hold_days
        self.dynamic_exit_start_day = dynamic_exit_start_day
        # 涨跌停阈值（百分比）：未指定时按 symbol/name 自动算
        if limit_pct is None:
            try:
                from data.limit_rules import get_limit_pct
                limit_pct = get_limit_pct(symbol, name)
            except Exception:
                limit_pct = 10.0
        self.limit_pct = float(limit_pct)
        self._validate()

    def _validate(self):
        required = ["trade_date", "open", "close", "high", "low", "pct_change"]
        missing = [c for c in required if c not in self.df.columns]
        if missing:
            raise ValueError(f"数据缺少必要列：{missing}")

        self.df["trade_date"] = pd.to_datetime(self.df["trade_date"])

        if not (0 < self.position_pct <= 1):
            raise ValueError("position_pct 必须在 (0, 1] 区间内")

        if not (0 < self.stop_loss_pct < 1):
            raise ValueError("stop_loss_pct 必须在 (0, 1) 区间内")

    def _calc_buy_commission(self, amount: float) -> float:
        """计算买入佣金（含最低5元）。"""
        return max(amount * BUY_COMMISSION, MIN_COMMISSION)

    def _calc_sell_cost(self, amount: float) -> tuple:
        """计算卖出佣金和印花税。返回 (commission, stamp_duty)。"""
        commission = max(amount * SELL_COMMISSION, MIN_COMMISSION)
        stamp_duty = amount * STAMP_DUTY
        return commission, stamp_duty

    def _should_dynamic_exit(self, df: pd.DataFrame, i: int) -> tuple:
        """
        动态时间评估：持仓满N天后，判断是否还有上涨动能。

        逻辑：
          有肉信号（任一满足→继续持有）：收阳、放量、价格上升
          没肉信号（任一满足→必须出场）：缩量收阴、当日跌超2%
          都不明显→保守出场

        参数：
            df : 含完整数据的 DataFrame
            i  : 当前行索引

        返回：
            (should_exit: bool, reason: str)
        """
        if i < 1:
            return True, "动态评估-数据不足"

        row = df.iloc[i]
        prev = df.iloc[i - 1]

        # ── 有肉的信号（任一满足 → 继续持有）──
        is_yang = row["close"] > row["open"]
        price_up = row["close"] > prev["close"]

        vol_up = False
        if "volume" in df.columns:
            try:
                vol_up = float(row["volume"]) > float(prev["volume"])
            except (ValueError, TypeError):
                pass

        has_momentum = is_yang or vol_up or price_up

        # ── 没肉的信号（任一满足 → 必须走）──
        is_yin = row["close"] < row["open"]

        vol_down = False
        if "volume" in df.columns:
            try:
                vol_down = float(row["volume"]) < float(prev["volume"])
            except (ValueError, TypeError):
                pass

        is_yin_shrink = is_yin and vol_down

        pct = 0.0
        if "pct_change" in df.columns:
            try:
                pct = float(row["pct_change"])
            except (ValueError, TypeError):
                pass
        big_drop = pct < -2.0

        must_exit = is_yin_shrink or big_drop

        # ── 判定 ──
        if must_exit:
            return True, "动态评估-动能衰竭"
        if has_momentum:
            return False, ""
        # 既无动能也无明显衰竭 → 保守出场
        return True, "动态评估-无动能"

    def run(self, signals: pd.Series) -> dict:
        """
        运行回测。

        参数：
            signals : 1=买入, -1=卖出, 0=无操作

        出场优先级：
            1. 硬止损（-5%）
            2. 跟踪止盈（冲高后回撤）
            3. 固定止盈（+4%，仅在跟踪未激活时）
            4. 策略卖出信号（-1）
            5. 动态时间评估（第N天起，有肉继续拿）
            6. 绝对持仓上限（第5天无论如何出场）
        """
        if len(signals) != len(self.df):
            raise ValueError("signals 长度必须与 df 长度一致")

        df = self.df.copy()
        df["signal"] = signals.values

        cash        = self.initial_cash
        shares      = 0
        position    = 0
        entry_price = 0.0
        entry_cost  = 0.0
        stop_price  = 0.0

        # ── 进阶出场状态 ──
        hold_days       = 0
        max_high        = 0.0
        trailing_active = False

        equity_curve = []
        trades       = []

        for i in range(len(df)):
            row = df.iloc[i]

            # ── 每日开盘前：更新持仓状态 ──
            if position == 1:
                hold_days += 1

                # 更新持仓期间最高价
                if row["high"] > max_high:
                    max_high = row["high"]

                # 检查是否激活跟踪止盈
                if (self.trailing_trigger_pct is not None
                        and not trailing_active
                        and max_high >= entry_price * (1 + self.trailing_trigger_pct)):
                    trailing_active = True
                    logger.debug(
                        "[%s] %s 跟踪止盈激活 最高=%.2f 触发线=%.2f",
                        self.symbol, row["trade_date"].date(),
                        max_high, entry_price * (1 + self.trailing_trigger_pct),
                    )

            # ── 记录每日权益 ──
            market_value = shares * row["close"]
            total_equity = cash + market_value

            equity_curve.append({
                "date"        : row["trade_date"],
                "cash"        : cash,
                "market_value": market_value,
                "total_equity": total_equity,
                "position"    : position,
            })

            # 最后一根 K 线没有下一个开盘价，无法执行次日交易
            if i >= len(df) - 1:
                continue

            next_row   = df.iloc[i + 1]
            signal     = row["signal"]
            exec_price = next_row["open"]

            # ── 涨跌停检测（按板块/ST 阈值，留 0.2% 余量避免精度误判）──
            open_pct      = (exec_price / row["close"] - 1) * 100
            limit_threshold = self.limit_pct - 0.2
            is_limit_up   = open_pct >= limit_threshold
            is_limit_down = open_pct <= -limit_threshold

            # ==========================================================
            #  卖出判断（按优先级，第一个命中就执行）
            # ==========================================================
            if position == 1:
                sell_reason = None

                # 优先级1：硬止损
                if row["close"] <= stop_price:
                    sell_reason = "硬止损"

                # 优先级2：跟踪止盈（已激活时，从最高价回落超过阈值）
                elif (trailing_active
                      and self.trailing_stop_pct is not None
                      and row["close"] < max_high * (1 - self.trailing_stop_pct)):
                    sell_reason = "跟踪止盈"

                # 优先级3：固定止盈（跟踪未激活时）
                elif (not trailing_active
                      and self.take_profit_pct is not None
                      and row["close"] >= entry_price * (1 + self.take_profit_pct)):
                    sell_reason = "止盈"

                # 优先级4：策略卖出信号
                elif signal == -1:
                    sell_reason = "卖出信号"

                # 优先级5：动态时间评估（有肉继续拿，没肉走）
                elif (self.dynamic_exit_start_day is not None
                      and hold_days >= self.dynamic_exit_start_day):
                    should_exit, reason = self._should_dynamic_exit(df, i)
                    if should_exit:
                        sell_reason = reason

                # 优先级6：绝对持仓上限
                elif (self.max_hold_days is not None
                      and hold_days >= self.max_hold_days):
                    sell_reason = "持仓到期"

                # ── 执行卖出 ──
                if sell_reason:
                    if is_limit_down:
                        logger.info(
                            "[%s] %s 跌停无法卖出（%s），继续持有",
                            self.symbol,
                            next_row["trade_date"].date(),
                            sell_reason,
                        )
                        continue

                    proceeds = shares * exec_price
                    commission, stamp_duty = self._calc_sell_cost(proceeds)
                    total_sell_cost = commission + stamp_duty
                    net_proceeds = proceeds - total_sell_cost

                    pnl     = net_proceeds - entry_cost
                    pnl_pct = pnl / entry_cost * 100 if entry_cost > 0 else 0.0

                    cash += net_proceeds

                    trades.append({
                        "date"      : next_row["trade_date"],
                        "action"    : "卖出",
                        "reason"    : sell_reason,
                        "price"     : round(exec_price, 4),
                        "shares"    : int(shares),
                        "commission": round(commission, 2),
                        "stamp_duty": round(stamp_duty, 2),
                        "cash_after": round(cash, 2),
                        "pnl"       : round(pnl, 2),
                        "pnl_pct"   : round(pnl_pct, 2),
                        "hold_days" : hold_days,
                    })

                    # 重置全部状态
                    shares          = 0
                    position        = 0
                    entry_price     = 0.0
                    entry_cost      = 0.0
                    stop_price      = 0.0
                    hold_days       = 0
                    max_high        = 0.0
                    trailing_active = False

                    continue  # 当天已卖出，不再处理买入

            # ==========================================================
            #  买入判断
            # ==========================================================
            if signal == 1 and position == 0:
                if is_limit_up:
                    logger.info(
                        "[%s] %s 涨停无法买入，跳过",
                        self.symbol, next_row["trade_date"].date(),
                    )
                    continue

                # 用 position_pct 比例的资金买入
                invest_cash = cash * self.position_pct
                max_shares = int(invest_cash / exec_price / 100) * 100

                if max_shares <= 0:
                    logger.warning(
                        "[%s] %s 资金不足(%.2f元)，无法买入100股(@%.2f元)",
                        self.symbol,
                        next_row["trade_date"].date(),
                        cash,
                        exec_price,
                    )
                    continue

                # 二次检查：扣除最低佣金后是否还够
                while max_shares > 0:
                    cost = max_shares * exec_price
                    commission = self._calc_buy_commission(cost)
                    total_cost = cost + commission
                    if total_cost <= cash:
                        break
                    max_shares -= 100

                if max_shares <= 0:
                    logger.warning(
                        "[%s] %s 扣除佣金后资金不足(@%.2f元)",
                        self.symbol,
                        next_row["trade_date"].date(),
                        exec_price,
                    )
                    continue

                cost       = max_shares * exec_price
                commission = self._calc_buy_commission(cost)
                total_cost = cost + commission

                cash       -= total_cost
                shares      = max_shares
                position    = 1
                entry_price = exec_price
                entry_cost  = total_cost
                stop_price  = entry_price * (1 - self.stop_loss_pct)

                # 初始化进阶出场状态
                hold_days       = 0    # 下一轮循环 +1 变成 1
                max_high        = entry_price
                trailing_active = False

                trades.append({
                    "date"      : next_row["trade_date"],
                    "action"    : "买入",
                    "reason"    : "买入信号",
                    "price"     : round(exec_price, 4),
                    "shares"    : int(shares),
                    "commission": round(commission, 2),
                    "stamp_duty": 0.0,
                    "cash_after": round(cash, 2),
                    "stop_price": round(stop_price, 4),
                })

        # ── 汇总结果 ──
        equity_df = pd.DataFrame(equity_curve)
        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()

        return {
            "equity_curve" : equity_df,
            "trades"       : trades_df,
            "final_equity" : equity_df["total_equity"].iloc[-1],
            "initial_cash" : self.initial_cash,
            "open_position": int(position),
            "open_shares"  : int(shares),
        }