# -*- coding: utf-8 -*-
"""回测引擎单元测试：成交时序、成本核算、现金约束、无未来函数"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np

from engine import BacktestEngine


def make_env():
    dates = pd.bdate_range("2024-01-01", periods=6)
    px = pd.DataFrame({"A": [10.0, 11.0, 12.0, 11.0, 13.0, 12.0]}, index=dates)
    opn = px.shift(1)  # 开盘=前收（简化测试环境）
    opn.iloc[0] = px.iloc[0]
    return opn, px


def test_signal_timing():
    """T日收盘的信号最早只能在T+1开盘成交：第0天不允许有任何成交"""
    opn, close = make_env()
    w = pd.DataFrame({"A": 1.0}, index=opn.index)   # 所有天都满仓信号
    eng = BacktestEngine(opn, close, commission=0.0, stamp=0.0, slippage=0.0)
    res = eng.run(w)
    t0 = res["trades"].iloc[0]
    assert t0["traded"] == 0.0, "第0日不应有成交（无前日信号）"
    assert t0["nav"] == eng.initial_cash
    # 第1日以 open[1]=close[0]=10 成交
    assert res["trades"].iloc[1]["traded"] > 0
    # 买入股数 = 现金/10
    assert abs(res["final_shares"]["A"] - eng.initial_cash / 10.0) < 1e-6
    print("PASS test_signal_timing")


def test_costs():
    """佣金双边收取且不低于最低5元；印花税仅卖出收取；现金永不透支"""
    opn, close = make_env()
    dates = opn.index
    # 第0天信号满仓，第1天信号清仓（第2天开盘卖出）
    w = pd.DataFrame({"A": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]}, index=dates)
    eng = BacktestEngine(opn, close, commission=0.001, stamp=0.0005,
                         slippage=0.0, min_commission=5.0)
    res = eng.run(w)
    buy_day = res["trades"].iloc[1]
    sell_day = res["trades"].iloc[2]
    # 买入日：费用 = max(佣金率×成交额, 5元)，且成交额+费用 ≤ 期初现金
    assert buy_day["traded"] > 0
    assert abs(buy_day["cost"] - max(0.001 * buy_day["traded"], 5.0)) < 1e-6
    assert buy_day["traded"] + buy_day["cost"] <= eng.initial_cash + 1e-6
    assert buy_day["cash"] >= -1e-6
    # 卖出日：费用 = 佣金 + 印花税（卖出全额）
    shares_after_buy = buy_day["traded"] / opn["A"].iloc[1]
    expected_sell = (shares_after_buy * opn["A"].iloc[2]) * (0.001 + 0.0005)
    assert abs(sell_day["cost"] - expected_sell) < 1e-3, (sell_day["cost"], expected_sell)
    assert sell_day["cash"] >= -1e-6
    print("PASS test_costs")


def test_no_negative_cash():
    """现金约束：任何时候现金不允许为负（买入以可用现金为限）"""
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2024-01-01", periods=120)
    px = pd.DataFrame({"A": 10 * np.exp(np.cumsum(rng.normal(0, 0.02, 120)))},
                      index=dates)
    opn = px.shift(1)
    opn.iloc[0] = px.iloc[0]
    w = pd.DataFrame({"A": rng.integers(0, 2, 120).astype(float)}, index=dates)
    eng = BacktestEngine(opn, px, commission=0.00025, stamp=0.0005, slippage=0.001)
    res = eng.run(w)
    assert (res["trades"]["cash"] >= -1e-6).all()
    assert (res["nav"] > 0).all()
    print("PASS test_no_negative_cash")


def test_no_lookahead_nav():
    """最后一天收盘的信号无法成交（没有次日），净值应全程不变——
    验证引擎绝不用当日信号当日成交（未来函数）"""
    opn, close = make_env()
    w = pd.DataFrame({"A": [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]}, index=opn.index)
    eng = BacktestEngine(opn, close, commission=0.0, stamp=0.0, slippage=0.0)
    res = eng.run(w)
    assert (res["trades"]["traded"] == 0).all(), "任何一天都不应有成交"
    assert (res["trades"]["nav"] == eng.initial_cash).all(), "净值应全程恒定期初值"
    print("PASS test_no_lookahead_nav")


def test_strategy_signals_use_past_only():
    """策略信号不得使用未来数据：截断历史应得到相同的早期信号"""
    from data_loader import fetch_local
    from strategies import ma_cross, bollinger_reversion
    df = fetch_local("510300")
    full = ma_cross(df["close"])
    cut = ma_cross(df["close"].loc[:"2023-12-31"])
    common = cut.index
    assert (full.loc[common, "asset"] == cut["asset"]).all(), "MA信号存在未来函数"
    full2 = bollinger_reversion(df["close"])
    cut2 = bollinger_reversion(df["close"].loc[:"2023-12-31"])
    assert (full2.loc[common, "asset"] == cut2["asset"]).all(), "布林信号存在未来函数"
    print("PASS test_strategy_signals_use_past_only")


if __name__ == "__main__":
    test_signal_timing()
    test_costs()
    test_no_negative_cash()
    test_no_lookahead_nav()
    test_strategy_signals_use_past_only()
    print("\nALL TESTS PASSED")
