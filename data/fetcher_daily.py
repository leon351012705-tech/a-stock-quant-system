"""
data/fetcher_daily.py
日常盘后更新专用 fetcher
主通道: Tushare daily()  — 1次API调用取全市场
备用:   BaoStock 逐股     — TCP协议，绕过HTTP代理问题
"""

import pandas as pd
import logging
import time
from datetime import datetime

logger = logging.getLogger(__name__)


# ========================================================
#  工具函数
# ========================================================

def _code_to_tushare(code: str) -> str:
    """'600000' → '600000.SH'"""
    code = code.strip().replace(" ", "")
    if code.startswith(("6", "9")):
        return f"{code}.SH"
    else:
        return f"{code}.SZ"


def _code_from_tushare(ts_code: str) -> str:
    """'600000.SH' → '600000'"""
    return ts_code.split(".")[0]


def _code_to_baostock(code: str) -> str:
    """'600000' → 'sh.600000'"""
    code = code.strip()
    if code.startswith(("6", "9")):
        return f"sh.{code}"
    else:
        return f"sz.{code}"


def _code_from_baostock(bs_code: str) -> str:
    """'sh.600000' → '600000'"""
    return bs_code.split(".")[1] if "." in bs_code else bs_code


def _date_to_tushare(date_str: str) -> str:
    """'2025-06-13' → '20250613'"""
    return date_str.replace("-", "")


def _date_from_tushare(date_str: str) -> str:
    """'20250613' → '2025-06-13'"""
    return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"


# ========================================================
#  Tushare 通道
# ========================================================

def fetch_all_via_tushare(token: str, trade_date: str) -> pd.DataFrame:
    """
    一次调用获取全市场当日日K线

    Parameters
    ----------
    token : str
        Tushare API token
    trade_date : str
        交易日期 'YYYY-MM-DD'

    Returns
    -------
    DataFrame: columns = [code, date, open, high, low, close, volume, amount,
                          pct_change, change]
               code = '600000' 格式
               date = 'YYYY-MM-DD' 格式
               volume = 股数
               amount = 元
    """
    import tushare as ts

    pro = ts.pro_api(token)
    ts_date = _date_to_tushare(trade_date)

    logger.info(f"[Tushare] 请求全市场日K: {trade_date}")

    # 一次拿全市场
    df = pro.daily(trade_date=ts_date)

    if df is None or df.empty:
        logger.warning(f"[Tushare] 返回空数据: {trade_date}")
        return pd.DataFrame()

    logger.info(f"[Tushare] 获取到 {len(df)} 条记录")

    # 标准化列名和格式
    result = pd.DataFrame()
    result["code"] = df["ts_code"].apply(_code_from_tushare)
    result["date"] = df["trade_date"].apply(_date_from_tushare)
    result["open"] = df["open"]
    result["high"] = df["high"]
    result["low"] = df["low"]
    result["close"] = df["close"]
    # tushare vol 单位是 手，转为股
    result["volume"] = df["vol"] * 100
    # tushare amount 单位是 千元，转为元
    result["amount"] = df["amount"] * 1000
    # 涨跌幅（tushare 列名是 pct_chg）
    result["pct_change"] = df["pct_chg"]
    # 涨跌额
    result["change"] = df["change"]

    return result


# ========================================================
#  BaoStock 通道
# ========================================================

class BaoStockFetcher:
    """BaoStock 日更新器（TCP协议，绕过HTTP代理问题）"""

    def __init__(self):
        self._logged_in = False

    def login(self):
        if not self._logged_in:
            import baostock as bs
            lg = bs.login()
            if lg.error_code != "0":
                raise ConnectionError(f"BaoStock 登录失败: {lg.error_msg}")
            self._logged_in = True
            logger.info("[BaoStock] 登录成功")

    def logout(self):
        if self._logged_in:
            import baostock as bs
            bs.logout()
            self._logged_in = False
            logger.info("[BaoStock] 已登出")

    def fetch_one(self, code: str, date: str) -> pd.DataFrame:
        """获取单只股票单日数据"""
        import baostock as bs
        self.login()

        bs_code = _code_to_baostock(code)

        rs = bs.query_history_k_data_plus(
            bs_code,
            "date,open,high,low,close,volume,amount,pctChg",
            start_date=date,
            end_date=date,
            frequency="d",
            adjustflag="2",  # 前复权
        )

        if rs.error_code != "0":
            logger.warning(f"[BaoStock] {code} 查询失败: {rs.error_msg}")
            return pd.DataFrame()

        rows = []
        while rs.next():
            rows.append(rs.get_row_data())

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows, columns=rs.fields)

        # 数值列转换
        for col in ["open", "high", "low", "close", "volume", "amount"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # 标准化列名
        if "pctChg" in df.columns:
            df["pctChg"] = pd.to_numeric(df["pctChg"], errors="coerce")
            df = df.rename(columns={"pctChg": "pct_change"})

        df.insert(0, "code", code)
        return df

    def fetch_all(self, stock_codes: list, date: str,
                  pause: float = 0.05) -> pd.DataFrame:
        """
        逐股获取当日数据。

        优化：先用 1-2 只股票探测，空就立即放弃，避免节假日空跑 15 分钟。

        Returns
        -------
        DataFrame: columns = [code, date, open, high, low, close, volume, amount, pct_change]
        """
        if not stock_codes:
            return pd.DataFrame()

        self.login()

        # ── 探测前哨：用 2 只大盘股先确认接口当日有数据 ──
        # 茅台 / 工行 几乎不会停牌；都为空 = 假期/数据未发布，无需查 5499 只
        probe_codes = [c for c in ["600519", "601398"] if c in stock_codes]
        if not probe_codes:
            probe_codes = stock_codes[:2]

        for pc in probe_codes:
            try:
                pdf = self.fetch_one(pc, date)
                if not pdf.empty:
                    break  # 任一只有数据就开干
            except Exception:
                continue
        else:
            # 全部探测都为空
            logger.warning(
                "[BaoStock] 探测 %s 均无数据 → 当日为非交易日或数据未发布，跳过全市场查询",
                "/".join(probe_codes),
            )
            self.logout()
            return pd.DataFrame()

        all_rows = []
        total = len(stock_codes)
        success = 0
        fail = 0

        logger.info(f"[BaoStock] 探测通过，开始逐股更新, 共 {total} 只")
        t0 = time.time()

        for i, code in enumerate(stock_codes):
            try:
                df = self.fetch_one(code, date)
                if not df.empty:
                    all_rows.append(df)
                    success += 1
                else:
                    fail += 1

                if (i + 1) % 500 == 0:
                    elapsed = time.time() - t0
                    eta = elapsed / (i + 1) * (total - i - 1)
                    logger.info(
                        f"[BaoStock] 进度 {i+1}/{total} "
                        f"成功={success} 失败={fail} "
                        f"预计剩余 {eta/60:.1f} 分钟"
                    )

                if pause > 0:
                    time.sleep(pause)

            except Exception as e:
                logger.warning(f"[BaoStock] {code} 异常: {e}")
                fail += 1

        self.logout()

        logger.info(
            f"[BaoStock] 完成: {success}/{total} 成功, "
            f"{fail} 失败, 耗时 {(time.time()-t0)/60:.1f} 分钟"
        )

        if all_rows:
            return pd.concat(all_rows, ignore_index=True)
        return pd.DataFrame()


# ========================================================
#  统一入口（自动切换）
# ========================================================

def fetch_daily_update(stock_codes: list, trade_date: str,
                       source: str = "auto",
                       tushare_token: str = "") -> pd.DataFrame:
    """
    日常更新统一入口

    Parameters
    ----------
    stock_codes : list
        股票代码列表 ['600000', '000001', ...]
    trade_date : str
        'YYYY-MM-DD'
    source : str
        'tushare' / 'baostock' / 'auto'
    tushare_token : str
        Tushare token

    Returns
    -------
    DataFrame: [code, date, open, high, low, close, volume, amount, pct_change, change]
    """

    result = pd.DataFrame()

    # —— 主通道: Tushare ——
    if source in ("tushare", "auto") and tushare_token:
        try:
            result = fetch_all_via_tushare(tushare_token, trade_date)
            if not result.empty:
                # 只保留我们关注的股票
                result = result[result["code"].isin(stock_codes)]
                logger.info(
                    f"[Tushare] 匹配到 {len(result)}/{len(stock_codes)} 只"
                )
                return result
        except Exception as e:
            logger.error(f"[Tushare] 失败，切换备用通道: {e}")

    # —— 备用通道: BaoStock ——
    if source in ("baostock", "auto"):
        try:
            bs_fetcher = BaoStockFetcher()
            result = bs_fetcher.fetch_all(stock_codes, trade_date)
            if not result.empty:
                return result
        except Exception as e:
            logger.error(f"[BaoStock] 也失败了: {e}")

    logger.error("所有数据通道均失败!")
    return result