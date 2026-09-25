# -*- coding: utf-8 -*-
"""数据获取与预处理模块

优先使用 akshare（东方财富/新浪接口）获取后复权日线行情，
失败时回退到 Yahoo Finance 公开接口。全部数据落盘为本地 CSV 缓存，
实验一律以本地缓存为准，避免接口变动影响结果的可复现性。
"""
import os
import time
import datetime as dt

import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)

START = "20190101"
END = dt.datetime.now().strftime("%Y%m%d")

# 时间序列策略标的：沪深300ETF(510300, 后复权)；横截面动量股票池（沪深300大市值样本）
ETF = {"510300": "沪深300ETF"}
STOCK_POOL = {
    "600519": "贵州茅台", "600036": "招商银行", "601318": "中国平安",
    "000858": "五粮液",   "000333": "美的集团", "300750": "宁德时代",
    "600030": "中信证券", "601012": "隆基绿能", "000651": "格力电器",
    "601888": "中国中免",
}


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """统一列名为英文小写，按日期升序、去重、去停牌空行"""
    rename = {"日期": "date", "开盘": "open", "收盘": "close", "最高": "high",
              "最低": "low", "成交量": "volume", "成交额": "amount",
              "Date": "date", "Open": "open", "Close": "close",
              "High": "high", "Low": "low", "Volume": "volume"}
    df = df.rename(columns=rename)
    df["date"] = pd.to_datetime(df["date"])
    df = df[["date", "open", "high", "low", "close", "volume"]]
    df = df.dropna().drop_duplicates(subset="date").sort_values("date")
    df = df.set_index("date")
    return df


def fetch_akshare_stock(symbol: str, start: str = START, end: str = END) -> pd.DataFrame:
    import akshare as ak
    df = ak.stock_zh_a_hist(symbol=symbol, period="daily",
                            start_date=start, end_date=end, adjust="hfq")
    return _clean(df)


def fetch_sina_stock(symbol: str, start: str = START, end: str = END) -> pd.DataFrame:
    """新浪接口备用源（eastmoney 限流时使用），后复权口径"""
    import akshare as ak
    prefix = "sh" if symbol.startswith(("6", "5")) else "sz"
    df = ak.stock_zh_a_daily(symbol=prefix + symbol, start_date=start,
                             end_date=end, adjust="hfq")
    return _clean(df)


def fetch_akshare_etf(symbol: str, start: str = START, end: str = END) -> pd.DataFrame:
    import akshare as ak
    df = ak.fund_etf_hist_em(symbol=symbol, period="daily",
                             start_date=start, end_date=end, adjust="hfq")
    return _clean(df)


def fetch_akshare_index(symbol: str, start: str = START, end: str = END) -> pd.DataFrame:
    import akshare as ak
    df = ak.index_zh_a_hist(symbol=symbol, period="daily",
                            start_date=start, end_date=end)
    return _clean(df)


def fetch_sina_index(symbol: str, start: str = START, end: str = END) -> pd.DataFrame:
    """新浪指数日线备用源（东财接口被限流/阻断时使用），指数无需复权"""
    import akshare as ak
    prefix = "sz" if symbol.startswith("399") else "sh"
    df = ak.stock_zh_index_daily(symbol=prefix + symbol)
    df = _clean(df)
    if len(df):
        df = df.loc[:end]
        df = df.loc[df.index >= pd.Timestamp(start)]
    return df


def _yahoo(symbol: str, start: str = START, end: str = END) -> pd.DataFrame:
    """Yahoo Finance chart 接口（无需鉴权），用于 akshare 失败时的备用数据源"""
    ysym = {"510300": "510300.SS", "000300": "000300.SS"}.get(
        symbol, symbol + (".SS" if symbol.startswith(("6", "5")) else ".SZ"))
    s = f"{start[:4]}-{start[4:6]}-{start[6:]}"
    e = f"{end[:4]}-{end[4:6]}-{end[6:]}"
    import urllib.request, json
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{ysym}"
           f"?period1={int(pd.Timestamp(s).timestamp())}"
           f"&period2={int(pd.Timestamp(e).timestamp())}&interval=1d"
           f"&events=div%2Csplit&includeAdjustedClose=true")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    data = json.loads(urllib.request.urlopen(req, timeout=30).read())
    res = data["chart"]["result"][0]
    q = res["indicators"]["quote"][0]
    adj = res["indicators"].get("adjclose", [{}])[0].get("adjclose")
    df = pd.DataFrame({
        "open": q["open"], "high": q["high"], "low": q["low"],
        "close": q["close"], "volume": q["volume"],
    }, index=pd.to_datetime(res["timestamp"], unit="ts" if False else "s").normalize())
    if adj is not None:
        factor = pd.Series(adj, index=df.index) / df["close"]
        for col in ("open", "high", "low", "close"):   # 全列后复权，与akshare口径一致
            df[col] = df[col] * factor
    df = df.dropna()
    return df


def fetch_local(symbol: str, kind: str = "stock") -> pd.DataFrame | None:
    path = os.path.join(DATA_DIR, f"{symbol}.csv")
    if os.path.exists(path):
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        return df if len(df) else None
    return None


def _with_retry(fn, n: int = 4, wait: float = 3.0):
    last = None
    for i in range(n):
        try:
            return fn()
        except Exception as e:            # 限速/断连时退避重试
            last = e
            time.sleep(wait * (i + 1))
    raise last


def get_data(symbol: str, kind: str = "stock", use_cache: bool = True) -> pd.DataFrame:
    """kind: stock / etf / index。带本地缓存与备用数据源"""
    if use_cache:
        cached = fetch_local(symbol)
        if cached is not None:
            return cached
    attempts = []
    if kind == "stock":
        attempts = [lambda: fetch_akshare_stock(symbol),
                    lambda: fetch_sina_stock(symbol)]
    elif kind == "etf":
        attempts = [lambda: fetch_akshare_etf(symbol)]
    elif kind == "index":
        attempts = [lambda: fetch_akshare_index(symbol),
                    lambda: fetch_sina_index(symbol)]
    attempts.append(lambda: _yahoo(symbol))
    last_err = None
    for fn in attempts:
        try:
            df = _with_retry(fn)
            if len(df) > 200:
                try:                      # 云端只读文件系统下跳过落盘，仅内存缓存
                    df.to_csv(os.path.join(DATA_DIR, f"{symbol}.csv"))
                except (OSError, PermissionError):
                    pass
                time.sleep(1.5)          # 接口限速保护
                return df
        except Exception as e:            # 换下一个数据源
            last_err = e
    raise RuntimeError(f"{symbol} 数据获取失败: {last_err}")


def load_universe() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """返回 (沪深300ETF日线, {代码: 股票日线})，按共同交易日对齐股票池"""
    etf = get_data("510300", kind="etf")
    pool = {}
    for sym, name in STOCK_POOL.items():
        pool[sym] = get_data(sym, kind="stock")
    return etf, pool


if __name__ == "__main__":
    etf, pool = load_universe()
    print("510300:", etf.index.min().date(), "->", etf.index.max().date(), len(etf), "rows")
    for s, df in pool.items():
        print(f"{s} {STOCK_POOL[s]}:", len(df), "rows",
              df.index.min().date(), "->", df.index.max().date())
