# -*- coding: utf-8 -*-
"""回测引擎

撮合规则（从机制上杜绝未来函数）：
  * 策略在 T 日收盘后产生目标权重 w(T)；
  * 引擎在 T+1 日以开盘价撮合（买入 open*(1+slippage)，卖出 open*(1-slippage)）；
  * 显式计入佣金（双边，含最低5元）与印花税（仅卖出，仅股票）；
  * 不支持杠杆与做空，目标权重非负且现金不允许为负。
"""
import pandas as pd
import numpy as np


class BacktestEngine:
    def __init__(self,
                 open_: pd.DataFrame,          # 日期×资产 开盘价（后复权）
                 close: pd.DataFrame,          # 日期×资产 收盘价（后复权）
                 commission: float = 0.00025,  # 佣金率（双边）
                 stamp: float = 0.0005,        # 印花税率（仅卖出）
                 slippage: float = 0.001,      # 单边滑点
                 min_commission: float = 5.0,  # 单笔最低佣金（元）
                 tradable: dict | None = None, # {资产: 是否股票(计印花税)}，默认全部视为股票
                 initial_cash: float = 1_000_000.0):
        assert (open_.index == close.index).all(), "open/close 日期未对齐"
        assert (open_.columns == close.columns).all(), "open/close 资产未对齐"
        self.open = open_.copy()
        self.close = close.copy()
        self.commission = commission
        self.stamp = stamp
        self.slippage = slippage
        self.min_commission = min_commission
        self.tradable = tradable or {c: True for c in close.columns}
        self.initial_cash = initial_cash
        self.dates = open_.index

    # ---------- 单资产交易执行 ----------
    def _fee(self, gross: float) -> float:
        """买入费用：佣金（双边，含最低5元）。佣金率为0时不收费"""
        if gross <= 0 or self.commission <= 0:
            return 0.0
        return max(self.commission * gross, self.min_commission)

    def _trade(self, asset: str, price: float, delta_value: float,
               cash: float, shares: float):
        """按目标市值变动执行交易，返回(现金变动后, 持股变动后, 交易成本, 成交额)"""
        cost = 0.0
        traded = 0.0
        if price <= 0 or np.isnan(price):
            return cash, shares, 0.0, 0.0
        if delta_value > 0:                       # 买入
            buy_price = price * (1 + self.slippage)
            spend = min(delta_value, cash)        # 现金约束：不允许透支
            fee = self._fee(spend)
            if spend + fee > cash:                # 给手续费预留空间，避免现金为负
                spend = max(cash - self._fee(cash), 0.0)
                fee = self._fee(spend)
            if spend <= 1.0 or spend + fee > cash:
                return cash, shares, 0.0, 0.0
            qty = spend / buy_price
            cash -= spend + fee
            shares += qty
            cost += fee
            traded += spend
        elif delta_value < 0:                     # 卖出
            sell_price = price * (1 - self.slippage)
            value = shares * sell_price
            sell_value = min(-delta_value, value)
            if sell_value <= 1.0:
                return cash, shares, 0.0, 0.0
            qty = sell_value / sell_price
            gross = qty * sell_price
            fee = self._fee(gross)
            if self.tradable.get(asset, True):
                fee += self.stamp * gross         # 印花税仅卖出、仅股票
            if fee >= gross:                      # 费用吃掉全部成交额，放弃该笔交易
                return cash, shares, 0.0, 0.0
            cash += gross - fee
            shares -= qty
            cost += fee
            traded += gross
        return cash, shares, cost, traded

    # ---------- 主循环 ----------
    def run(self, target_w: pd.DataFrame) -> dict:
        """target_w: 日期×资产，w[t] 为 T 日收盘后决定的目标权重（0~1，行和<=1）"""
        w = target_w.reindex(self.dates).fillna(0.0).clip(lower=0.0)
        cash = self.initial_cash
        shares = pd.Series(0.0, index=self.open.columns)
        rows = []
        pending = None                             # 上一日收盘决定的目标权重
        last_target = None                         # 最近一次已执行的目标权重
        total_cost = 0.0
        total_traded = 0.0

        for i, t in enumerate(self.dates):
            o = self.open.loc[t]
            c = self.close.loc[t]
            # 1) 开盘撮合昨日信号：仅当目标权重发生变化时才交易（触发式调仓，
            #    持有期间仓位随价格自然漂移，不产生每日再平衡的虚假换手）
            day_cost = 0.0
            day_traded = 0.0
            if pending is not None and i > 0:
                changed = (last_target is None or
                           not np.allclose(pending.values,
                                           last_target.values, atol=1e-9))
                if changed:
                    nav_open = cash + float((shares * o.fillna(0)).sum())
                    if nav_open > 0:
                        for a in self.open.columns:
                            cur = float(shares[a] * (o[a] if not np.isnan(o[a]) else 0))
                            want = float(pending[a]) * nav_open
                            cash, shares[a], fee, tr = self._trade(
                                a, o[a] if not np.isnan(o[a]) else np.nan,
                                want - cur, cash, float(shares[a]))
                            day_cost += fee
                            day_traded += tr
                    last_target = pending.copy()
            total_cost += day_cost
            total_traded += day_traded
            # 2) 收盘估值
            nav_close = cash + float((shares * c.fillna(0)).sum())
            rows.append({"date": t, "cash": cash, "nav": nav_close,
                         "cost": day_cost, "traded": day_traded,
                         "n_pos": int((shares > 0).sum())})
            # 3) 收盘后生成次日目标
            pending = w.loc[t]
        nav = pd.Series([r["nav"] for r in rows], index=self.dates)
        trades = pd.DataFrame(rows).set_index("date")
        trades["ret"] = nav.pct_change().fillna(0.0)
        # 年化换手率：日均双边成交额/净值 ×252
        turnover = float((trades["traded"] / trades["nav"]).replace(
            np.inf, np.nan).dropna().mean() * 252) if len(trades) else 0.0
        return {"nav": nav, "trades": trades,
                "total_cost": total_cost, "total_traded": total_traded,
                "annual_turnover": turnover,
                "final_cash": cash,
                "final_shares": shares}
