# -*- coding: utf-8 -*-
"""实验主脚本：三类策略 + 基准的对比实验、成本敏感性、样本外检验

样本内(IS)：2019-01-01 ~ 2023-12-31；样本外(OOS)：2024-01-01 ~ 今
输出：results/ 下的指标CSV与图（论文第5章使用）
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import load_universe
from engine import BacktestEngine
from strategies import (buy_and_hold, ma_cross, bollinger_reversion,
                        cross_sectional_momentum, vol_target_scale)
import performance as perf

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS, exist_ok=True)

OOS_START = "2024-01-01"

# 成本参数：佣金万2.5（双边，最低5元）、印花税卖出0.05%、滑点0.1%
COMMISSION, STAMP, SLIPPAGE = 0.00025, 0.0005, 0.001


def seg(name, nav, tag):
    s = perf.segment_stats(nav)
    return {"策略": name, "区间": tag, **{k: round(v, 4) if v == v else np.nan
                                        for k, v in s.items()}}


def run_single(name, sig, etf):
    """单标的策略回测（ETF不计印花税）"""
    dates = etf.index
    open_, close = etf[["open"]], etf[["close"]]
    open_.columns = close.columns = ["ETF"]
    eng = BacktestEngine(open_, close, COMMISSION, STAMP, SLIPPAGE,
                         tradable={"ETF": False})
    w = sig.reindex(dates).fillna(0.0)
    w.columns = ["ETF"]
    res = eng.run(w)
    res["name"] = name
    return res


def main():
    etf_raw, pool_raw = load_universe()
    print(f"510300: {etf_raw.index[0].date()} ~ {etf_raw.index[-1].date()}  {len(etf_raw)}日")

    # ---------- 数据层图示 ----------
    perf.plot_price_with_indicators(
        etf_raw["close"], os.path.join(RESULTS, "fig4_1_price.png"),
        "沪深300ETF日线走势与技术指标（2019.1—2026.9，后复权）")

    # ---------- 实验1/2：ETF上的时序策略 ----------
    sig_ma = ma_cross(etf_raw["close"], 5, 20)
    sig_boll = bollinger_reversion(etf_raw["close"], 20, 2.0)
    sig_bh = buy_and_hold(["ETF"], etf_raw.index)

    res_ma = run_single("双均线", sig_ma, etf_raw)
    res_boll = run_single("均值回归", sig_boll, etf_raw)
    res_bh = run_single("买入持有", sig_bh, etf_raw)

    # ---------- 实验3：横截面动量（10股池） ----------
    common_idx = None
    for df in pool_raw.values():
        idx = df.index
        common_idx = idx if common_idx is None else common_idx.intersection(idx)
    common_idx = common_idx.intersection(etf_raw.index)
    pool_close = pd.DataFrame({s: pool_raw[s]["close"].reindex(common_idx)
                               for s in pool_raw})
    pool_open = pd.DataFrame({s: pool_raw[s]["open"].reindex(common_idx)
                              for s in pool_raw})

    sig_mom = cross_sectional_momentum(pool_close, 20, 3, "M")
    sig_poolbh = buy_and_hold(list(pool_close.columns), pool_close.index)

    eng_mom = BacktestEngine(pool_open, pool_close, COMMISSION, STAMP, SLIPPAGE,
                             tradable={s: True for s in pool_close.columns})
    res_mom = eng_mom.run(sig_mom)
    res_mom["name"] = "动量Top3"
    eng_poolbh = BacktestEngine(pool_open, pool_close, COMMISSION, STAMP, SLIPPAGE,
                                tradable={s: True for s in pool_close.columns})
    res_poolbh = eng_poolbh.run(sig_poolbh)
    res_poolbh["name"] = "股票池等权持有"

    # ETF基准对齐到共同交易日
    etf_common = etf_raw.reindex(common_idx)
    res_bh_common = run_single("买入持有", buy_and_hold(["ETF"], common_idx), etf_common)

    # ---------- 绩效汇总 ----------
    full = {"双均线(5,20)": res_ma, "均值回归(20,2σ)": res_boll,
            "买入持有基准": res_bh}
    rows = []
    for name, r in full.items():
        rows.append(seg(name, r["nav"], "全样本"))
        rows.append(seg(name, r["nav"].loc[:OOS_START], "样本内"))
        rows.append(seg(name, r["nav"].loc[OOS_START:], "样本外"))
    for name, r in {"动量Top3": res_mom, "股票池等权持有": res_poolbh,
                    "买入持有基准(对齐)": res_bh_common}.items():
        rows.append(seg(name, r["nav"], "全样本"))
        rows.append(seg(name, r["nav"].loc[:OOS_START], "样本内"))
        rows.append(seg(name, r["nav"].loc[OOS_START:], "样本外"))
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(RESULTS, "metrics.csv"), index=False, encoding="utf-8-sig")
    print("\n===== 绩效汇总 =====")
    print(table.to_string(index=False))

    # 换手与成本
    cost_rows = []
    for name, r in [("双均线(5,20)", res_ma), ("均值回归(20,2σ)", res_boll),
                    ("动量Top3", res_mom), ("买入持有基准", res_bh)]:
        cost_rows.append({"策略": name, "年化换手率": round(r["annual_turnover"], 2),
                          "总交易成本(元)": round(r["total_cost"], 0),
                          "总成交额(元)": round(r["total_traded"], 0)})
    pd.DataFrame(cost_rows).to_csv(os.path.join(RESULTS, "costs.csv"),
                                   index=False, encoding="utf-8-sig")
    print("\n===== 换手与成本 =====")
    print(pd.DataFrame(cost_rows).to_string(index=False))

    # ---------- 成本敏感性（双均线） ----------
    levels = {"零成本(0%)": 0.0, "半成本(50%)": 0.5, "实际成本(100%)": 1.0}
    w_ma = sig_ma.reindex(etf_raw.index).fillna(0.0)
    w_ma.columns = ["ETF"]
    sens_rows = []
    for lv, mult in levels.items():
        eng = BacktestEngine(etf_raw[["open"]].rename(columns={"open": "ETF"}),
                             etf_raw[["close"]].rename(columns={"close": "ETF"}),
                             COMMISSION * mult, STAMP * mult, SLIPPAGE * mult,
                             tradable={"ETF": False})
        r = eng.run(w_ma)
        for tag, nav in [("样本内", r["nav"].loc[:OOS_START]),
                         ("样本外", r["nav"].loc[OOS_START:])]:
            s = perf.segment_stats(nav)
            sens_rows.append({"成本情景": lv, "区间": tag,
                              "年化收益率": round(s["年化收益率"], 4),
                              "夏普比率": round(s["夏普比率"], 3),
                              "最大回撤": round(s["最大回撤"], 4)})
    sens = pd.DataFrame(sens_rows)
    sens.to_csv(os.path.join(RESULTS, "cost_sensitivity.csv"), index=False,
                encoding="utf-8-sig")
    print("\n===== 成本敏感性（双均线） =====")
    print(sens.to_string(index=False))

    # ---------- 图 ----------
    perf.plot_nav({k: v["nav"] for k, v in full.items()},
                  "时序策略净值对比（沪深300ETF，全样本）",
                  os.path.join(RESULTS, "fig5_1_nav_etf.png"), OOS_START)
    perf.plot_drawdown({k: v["nav"] for k, v in full.items()},
                       "时序策略回撤对比（沪深300ETF）",
                       os.path.join(RESULTS, "fig5_2_dd_etf.png"))
    perf.plot_nav({"动量Top3": res_mom["nav"],
                   "股票池等权持有": res_poolbh["nav"],
                   "买入持有基准": res_bh_common["nav"]},
                  "横截面动量策略净值对比（全样本）",
                  os.path.join(RESULTS, "fig5_3_nav_mom.png"), OOS_START)

    piv = sens.pivot_table(index="成本情景", columns="区间", values="年化收益率")
    piv = piv.reindex(["零成本(0%)", "半成本(50%)", "实际成本(100%)"])
    perf.plot_bars(piv, "双均线策略收益对交易成本的敏感性",
                   os.path.join(RESULTS, "fig5_4_cost.png"), "年化收益率")

    # ---------- 仓位控制实验：波动率目标（Moreira & Muir, 2017） ----------
    TV = 0.15
    w_base = sig_ma.reindex(etf_raw.index).fillna(0.0)
    scale = vol_target_scale(etf_raw["close"], TV).reindex(etf_raw.index).fillna(0.0)
    sig_vt = w_base.mul(scale, axis=0)
    res_vt = run_single("双均线+波动率目标", sig_vt, etf_raw)
    # 动量组合的波动率目标：以股票池等权组合的已实现波动率作为整体风险估计
    eq_close = (pool_close.pct_change().fillna(0).mean(axis=1) + 1).cumprod()
    scale_pool = vol_target_scale(eq_close, TV).reindex(common_idx).fillna(0.0)
    sig_mom_vt = sig_mom.mul(scale_pool, axis=0)
    res_mom_vt = eng_mom.run(sig_mom_vt)
    res_mom_vt["name"] = "动量Top3+波动率目标"
    vol_rows = []
    for name, r in [("满仓双均线", res_ma), ("波动率目标双均线(15%)", res_vt),
                    ("买入持有基准", res_bh),
                    ("满仓动量Top3", res_mom), ("波动率目标动量Top3(15%)", res_mom_vt)]:
        for tag, nav in [("全样本", r["nav"]), ("样本内", r["nav"].loc[:OOS_START]),
                         ("样本外", r["nav"].loc[OOS_START:])]:
            s = perf.segment_stats(nav)
            vol_rows.append({"策略": name, "区间": tag,
                             "年化收益率": round(s["年化收益率"], 4),
                             "夏普比率": round(s["夏普比率"], 3),
                             "最大回撤": round(s["最大回撤"], 4)})
    vol_tab = pd.DataFrame(vol_rows)
    vol_tab.to_csv(os.path.join(RESULTS, "metrics_voltarget.csv"),
                   index=False, encoding="utf-8-sig")
    print("\n===== 仓位控制实验（波动率目标15%，0.1步长分档） =====")
    print(vol_tab.to_string(index=False))
    perf.plot_nav({"满仓双均线": res_ma["nav"],
                   "波动率目标双均线": res_vt["nav"],
                   "买入持有基准": res_bh["nav"]},
                  "波动率目标仓位的效果（全样本，期初=1）",
                  os.path.join(RESULTS, "fig5_5_nav_voltarget.png"), OOS_START)
    perf.plot_nav({"满仓动量Top3": res_mom["nav"],
                   "波动率目标动量Top3": res_mom_vt["nav"],
                   "股票池等权持有": res_poolbh["nav"]},
                  "动量组合波动率目标仓位的效果（全样本，期初=1）",
                  os.path.join(RESULTS, "fig5_6_nav_mom_vt.png"), OOS_START)

    # 净值序列存档
    pd.DataFrame({"双均线": res_ma["nav"], "均值回归": res_boll["nav"],
                  "买入持有": res_bh["nav"]}).to_csv(
        os.path.join(RESULTS, "nav_etf.csv"), encoding="utf-8-sig")
    pd.DataFrame({"动量Top3": res_mom["nav"], "股票池等权": res_poolbh["nav"],
                  "买入持有(对齐)": res_bh_common["nav"]}).to_csv(
        os.path.join(RESULTS, "nav_mom.csv"), encoding="utf-8-sig")

    # 每期动量持仓记录（论文示例用）
    held = (sig_mom > 0).sum(axis=1)
    pd.DataFrame({"持仓数": held}).to_csv(
        os.path.join(RESULTS, "mom_positions.csv"), encoding="utf-8-sig")

    print("\n实验完成，图表与CSV已输出至 results/")


if __name__ == "__main__":
    main()
