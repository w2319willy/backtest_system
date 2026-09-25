# -*- coding: utf-8 -*-
"""技术指标计算（仅依赖截至当日的信息，无未来函数）"""
import pandas as pd
import numpy as np


def sma(close: pd.Series, n: int) -> pd.Series:
    return close.rolling(n, min_periods=n).mean()


def bollinger(close: pd.Series, n: int = 20, k: float = 2.0):
    """返回(中轨, 下轨, 上轨)"""
    ma = close.rolling(n, min_periods=n).mean()
    sd = close.rolling(n, min_periods=n).std(ddof=0)
    return ma, ma - k * sd, ma + k * sd


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    diff = close.diff()
    up = diff.clip(lower=0).rolling(n).mean()
    dn = (-diff.clip(upper=0)).rolling(n).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(),
                    (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def momentum_return(close: pd.Series, n: int = 20) -> pd.Series:
    """过去n日收益率（含当日收盘），用于横截面排序"""
    return close / close.shift(n) - 1
