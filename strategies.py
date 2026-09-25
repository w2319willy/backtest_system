# -*- coding: utf-8 -*-
"""策略模块：信号一律只使用截至 T 日收盘的信息，由引擎在 T+1 开盘撮合"""
import pandas as pd
import numpy as np

from indicators import sma, bollinger, momentum_return


def buy_and_hold(universe: list[str], dates: pd.DatetimeIndex) -> pd.DataFrame:
    """期初买入并持有，作为业绩基准（全程保持满仓目标，由触发式引擎保证仅成交一次）"""
    w = pd.DataFrame(1.0 / len(universe), index=dates, columns=universe)
    return w


def ma_cross(close: pd.Series, fast: int = 5, slow: int = 20) -> pd.DataFrame:
    """双均线趋势跟踪：MA_fast 上穿 MA_slow 持有，下穿空仓"""
    wf, ws = sma(close, fast), sma(close, slow)
    sig = (wf > ws).astype(float)
    sig[wf.isna() | ws.isna()] = 0.0
    return sig.to_frame("asset")


def bollinger_reversion(close: pd.Series, n: int = 20, k: float = 2.0) -> pd.DataFrame:
    """布林带均值回归：跌破下轨买入，回升至中轨卖出（状态机实现）"""
    ma, lo, hi = bollinger(close, n, k)
    pos = 0.0
    out = []
    for t in close.index:
        c, m, l = close.loc[t], ma.loc[t], lo.loc[t]
        if not np.isnan(m):
            if pos == 0.0 and not np.isnan(l) and c < l:
                pos = 1.0
            elif pos == 1.0 and c >= m:
                pos = 0.0
        out.append(pos)
    return pd.Series(out, index=close.index, name="asset").to_frame()


def cross_sectional_momentum(pool_close: pd.DataFrame, lookback: int = 20,
                             top: int = 3,
                             rebalance: str = "M") -> pd.DataFrame:
    """横截面动量：每月末按过去lookback日收益排序，等权持有前top名

    月末最后一个交易日收盘产生信号，次月首个交易日开盘调仓（由引擎保证）。
    """
    mom = momentum_return(pool_close, lookback)
    reb_dates = mom.groupby(mom.index.to_period(rebalance)).apply(
        lambda g: g.index[-1])
    w = pd.DataFrame(0.0, index=pool_close.index, columns=pool_close.columns)
    cur = []
    for t in pool_close.index:
        if t in set(reb_dates):
            scores = mom.loc[t].dropna()
            if len(scores) >= top:
                cur = scores.nlargest(top).index.tolist()
        for a in cur:
            w.loc[t, a] = 1.0 / top
    return w
