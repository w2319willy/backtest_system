# -*- coding: utf-8 -*-
"""量化交易策略回测系统 - Web交互界面（Streamlit）

运行方式：
    cd 回测系统
    python -m streamlit run app.py --server.port 8501
浏览器访问 http://localhost:8501

界面仅做参数收集与结果呈现，回测计算完全复用 engine/strategies/performance 模块。
支持输入任意A股代码（股票/ETF/指数）：按需联网获取后复权日线并落盘缓存，
可强制刷新至最新交易日；动量组合支持自定义股票池。
"""
import os
import re
import sys
import datetime as dt

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import STOCK_POOL, get_data
from engine import BacktestEngine
from strategies import (buy_and_hold, cross_sectional_momentum,
                        bollinger_reversion, ma_cross)
import performance as perf

st.set_page_config(page_title="量化交易策略回测系统", page_icon="📈", layout="wide")

KIND_CN = {"stock": "股票", "etf": "ETF", "index": "指数"}
KIND_CHOICES = ["自动识别", "股票", "ETF", "指数"]
CODE_RE = re.compile(r"^\d{6}$")
DEFAULT_POOL = " ".join(STOCK_POOL)


def infer_kind(code: str, choice: str) -> str:
    """由品种选择与代码前缀推断数据类别：5/1开头为ETF，其余默认股票；指数需显式选择"""
    if choice != "自动识别":
        return {"股票": "stock", "ETF": "etf", "指数": "index"}[choice]
    return "etf" if code.startswith(("5", "1")) else "stock"


def _sina_name(code: str, kind: str) -> str:
    """新浪行情接口查证券简称（东财名称接口不可用时的备用通道）"""
    import urllib.request
    if kind == "index":
        sym = ("s_sz" if code.startswith("399") else "s_sh") + code
    else:
        sym = ("sh" if code.startswith(("6", "5")) else "sz") + code
    req = urllib.request.Request(f"https://hq.sinajs.cn/list={sym}",
                                 headers={"Referer": "https://finance.sina.com.cn"})
    data = urllib.request.urlopen(req, timeout=8).read().decode("gbk")
    payload = data.split('"')[1]
    return payload.split(",")[0] if payload else ""


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_name(code: str, kind: str) -> str:
    """查询证券简称（尽力而为：东财接口→新浪接口→退回代码本身）"""
    try:
        import akshare as ak
        if kind == "stock":
            info = ak.stock_individual_info_em(symbol=code)
            d = dict(zip(info["item"], info["value"]))
            name = str(d.get("股票简称", ""))
            if name:
                return name
        elif kind == "etf":
            lst = ak.fund_etf_spot_em()
            row = lst[lst["代码"] == code]
            if len(row):
                return str(row.iloc[0]["名称"])
        elif kind == "index":
            lst = ak.stock_zh_index_spot_em(symbol="沪深重要指数")
            row = lst[lst["代码"] == code]
            if len(row):
                return str(row.iloc[0]["名称"])
    except Exception:
        pass
    try:
        name = _sina_name(code, kind)
        if name:
            return name
    except Exception:
        pass
    return code


@st.cache_data(show_spinner="正在获取行情数据...")
def load_price(symbol: str, kind: str, refresh: bool = False) -> pd.DataFrame:
    return get_data(symbol, kind=kind, use_cache=not refresh)


# ---------------- 数据与回测（带缓存） ----------------
@st.cache_data(show_spinner="正在回测...")
def run_backtest(symbol: str, kind: str, strategy: str, params: tuple,
                 commission: float, stamp: float, slippage: float,
                 cash0: float, start: str, end: str, refresh: bool = False) -> dict:
    df = load_price(symbol, kind, refresh).loc[start:end]
    if len(df) < 60:
        raise ValueError("所选区间数据不足（不足60个交易日），请检查代码或调整区间")
    dates = df.index
    open_ = df[["open"]].rename(columns={"open": "asset"})
    close = df[["close"]].rename(columns={"close": "asset"})
    # 印花税仅对股票征收；ETF/指数不缴
    eng = BacktestEngine(open_, close, commission, stamp, slippage,
                         tradable={"asset": kind == "stock"}, initial_cash=cash0)

    if strategy == "双均线趋势跟踪":
        fast, slow = params
        sig = ma_cross(df["close"], fast, slow)
        sig.columns = ["asset"]
    elif strategy == "布林带均值回归":
        n, k = params
        sig = bollinger_reversion(df["close"], int(n), float(k))
        sig.columns = ["asset"]
    elif strategy == "买入持有基准":
        sig = buy_and_hold(["asset"], dates)
    else:
        raise ValueError("单标的模式不支持该策略")
    w = sig.reindex(dates).fillna(0.0)
    res = eng.run(w)
    # 基准：同标的买入持有（相同成本口径）
    res["bench"] = eng.run(buy_and_hold(["asset"], dates))
    res["actual_start"] = dates[0]
    res["actual_end"] = dates[-1]
    res["n_days"] = len(dates)
    return res


@st.cache_data(show_spinner="正在获取股票池行情并回测...")
def run_momentum(pool_text: str, params: tuple, commission: float, stamp: float,
                 slippage: float, cash0: float, start: str, end: str,
                 refresh: bool = False) -> dict:
    codes = list(dict.fromkeys(re.findall(r"\d{6}", pool_text)))
    if len(codes) < 3:
        raise ValueError("股票池至少需要3个有效的6位代码")
    pool, skipped = {}, []
    for c in codes:
        kind = infer_kind(c, "自动识别")
        try:
            pool[c] = load_price(c, kind, refresh).loc[start:end]
        except Exception:
            skipped.append(c)
    if len(pool) < 3:
        raise ValueError(f"有效标的不足3个（失败：{'、'.join(skipped) or '无'}），无法构建股票池")
    common = None
    for df in pool.values():
        common = df.index if common is None else common.intersection(df.index)
    pc = pd.DataFrame({s: pool[s]["close"].reindex(common) for s in pool})
    po = pd.DataFrame({s: pool[s]["open"].reindex(common) for s in pool})
    eng = BacktestEngine(po, pc, commission, stamp, slippage,
                         tradable={s: infer_kind(s, "自动识别") == "stock" for s in pc},
                         initial_cash=cash0)
    lookback, top = params
    sig = cross_sectional_momentum(pc, lookback, top, "M")
    res = eng.run(sig)
    res["bench"] = eng.run(buy_and_hold(list(pc.columns), pc.index))
    res["skipped"] = skipped
    res["pool_size"] = len(pc.columns)
    res["actual_start"] = common[0]
    res["actual_end"] = common[-1]
    res["n_days"] = len(common)
    return res


def seg_stats(nav: pd.Series, oos_start) -> dict:
    full = perf.segment_stats(nav)
    out = {"全样本": full}
    if oos_start and nav.index.min() < oos_start <= nav.index.max():
        out["样本内"] = perf.segment_stats(nav.loc[:oos_start])
        out["样本外"] = perf.segment_stats(nav.loc[oos_start:])
    return out


def fmt_pct(x):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x * 100:+.2f}%"


def fmt_num(x, nd=2):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


# ---------------- 侧边栏：参数 ----------------
st.sidebar.title("⚙️ 回测设置")

mode = st.sidebar.radio("回测模式", ["单标的策略", "动量组合（自选股票池）"], horizontal=True)

if mode == "单标的策略":
    c1, c2 = st.sidebar.columns([1.2, 1])
    code = c1.text_input("证券代码（6位）", value="510300",
                         help="示例：600519贵州茅台、510300沪深300ETF；指数（如000300上证指数、399006创业板指）请把类型选为“指数”")
    kind_choice = c2.selectbox("品种类型", KIND_CHOICES)
    code = code.strip()
    if not CODE_RE.match(code):
        st.sidebar.error("请输入6位数字代码，例如 600519")
        st.stop()
    kind = infer_kind(code, kind_choice)
    strategy = st.sidebar.selectbox("选择策略", ["双均线趋势跟踪", "布林带均值回归", "买入持有基准"])
    if strategy == "双均线趋势跟踪":
        c1, c2 = st.sidebar.columns(2)
        fast = c1.slider("短期均线", 3, 30, 5)
        slow = c2.slider("长期均线", 10, 120, 20)
        if fast >= slow:
            st.sidebar.warning("短期均线应小于长期均线，已自动调整")
            fast, slow = min(fast, slow - 1), slow
        params = (fast, slow)
        st.sidebar.caption(f"当前参数：MA{fast} / MA{slow}，金叉买入、死叉卖出")
    elif strategy == "布林带均值回归":
        n = st.sidebar.slider("均线周期N", 10, 60, 20)
        k = st.sidebar.slider("标准差倍数K", 1.0, 3.0, 2.0, 0.1)
        params = (n, k)
        st.sidebar.caption(f"当前参数：布林带({n}, {k}σ)，跌破下轨买入、回升中轨卖出")
    else:
        params = ()
else:
    code, kind = None, None
    strategy = "横截面动量"
    pool_text = st.sidebar.text_area("股票池代码（6位，空格/逗号分隔）", value=DEFAULT_POOL,
                                     help="可输入任意A股代码，获取失败的标的将被自动跳过；建议至少5只")
    c1, c2 = st.sidebar.columns(2)
    lookback = c1.slider("动量回看期", 10, 60, 20)
    top = c2.slider("持有只数", 1, 5, 3)
    params = (lookback, top)
    st.sidebar.caption(f"每月末按过去{lookback}日收益排序，等权持有最强{top}只")

st.sidebar.divider()
st.sidebar.subheader("交易成本与资金")
commission = st.sidebar.slider("佣金率（‱，双边）", 0.0, 10.0, 2.5, 0.5) / 10000
stamp = st.sidebar.slider("印花税率（‱，卖出）", 0.0, 20.0, 5.0, 0.5) / 10000
slippage = st.sidebar.slider("滑点（‱，单边）", 0.0, 20.0, 10.0, 0.5) / 10000
cash0 = st.sidebar.select_slider("初始资金（元）", [100_000, 500_000, 1_000_000, 5_000_000],
                                 value=1_000_000)

st.sidebar.subheader("回测区间")
c1, c2 = st.sidebar.columns(2)
start_date = c1.date_input("开始", dt.date(2019, 1, 2),
                           min_value=dt.date(2015, 1, 1), max_value=dt.date.today())
end_date = c2.date_input("结束", dt.date.today(),
                         min_value=dt.date(2015, 1, 1), max_value=dt.date.today())
oos_date = st.sidebar.date_input("样本外起点（IS/OOS分界）", dt.date(2024, 1, 1),
                                 min_value=dt.date(2015, 1, 1), max_value=dt.date.today())
refresh = st.sidebar.checkbox("强制刷新行情", value=False,
                              help="勾选后忽略本地缓存，重新联网下载该代码的最新日线数据")

run_clicked = st.sidebar.button("🚀 运行回测", type="primary", use_container_width=True)
st.sidebar.divider()
st.sidebar.caption("撮合规则：T日收盘产生信号，T+1日开盘价成交；"
                   "成本含佣金（最低5元）、印花税与滑点；不支持杠杆与做空。")

# ---------------- 主区域 ----------------
st.title("📈 基于Python的量化交易策略回测系统")
st.caption("华南理工大学辅修学士学位毕业设计 · 数据层—引擎层—策略层—评估层 · "
           "支持任意A股代码按需获取")

if not run_clicked:
    st.info("请在左侧设置参数后，点击「🚀 运行回测」。输入任意6位A股代码即可联网获取行情。")
    st.stop()

try:
    if mode == "单标的策略":
        res = run_backtest(code, kind, strategy, params, commission, stamp,
                           slippage, float(cash0), str(start_date), str(end_date),
                           refresh)
        name = fetch_name(code, kind) if not refresh else fetch_name(code, kind)
        title_asset = f"{name}({code})｜{KIND_CN[kind]}"
    else:
        res = run_momentum(pool_text, params, commission, stamp, slippage,
                           float(cash0), str(start_date), str(end_date), refresh)
        title_asset = f"自选股票池（有效{res['pool_size']}只）"
except Exception as e:
    st.error(f"回测失败：{e}")
    st.caption("提示：请检查证券代码是否正确；指数代码（如000300、399006）需把品种类型选为“指数”；"
               "新代码首次获取需要联网，约需数秒。")
    st.stop()

nav = res["nav"].dropna()
bench_nav = res["bench"]["nav"].reindex(nav.index).dropna()
bench_label = "买入持有" if mode == "单标的策略" else "股票池等权持有"
oos_ts = pd.Timestamp(oos_date) if start_date < oos_date < end_date else None

st.caption(f"标的：{title_asset}｜实际区间：{res['actual_start'].date()} ~ "
           f"{res['actual_end'].date()}（{res['n_days']}个交易日）"
           + (f"｜跳过无法获取：{'、'.join(res['skipped'])}" if res.get("skipped") else ""))

# 指标卡片
stats_all = perf.segment_stats(nav)
stats_oos = (perf.segment_stats(nav.loc[oos_ts:]) if oos_ts else None)
cols = st.columns(6)
cols[0].metric("全样本年化收益", fmt_pct(stats_all.get("年化收益率")))
cols[1].metric("样本外年化收益", fmt_pct(stats_oos.get("年化收益率")) if stats_oos else "—")
cols[2].metric("全样本夏普比率", fmt_num(stats_all.get("夏普比率")))
cols[3].metric("最大回撤", fmt_pct(stats_all.get("最大回撤")))
cols[4].metric("年化换手率", f"{res['annual_turnover']:.1f} 倍")
cols[5].metric("总交易成本", f"{res['total_cost']:,.0f} 元")

st.divider()

# 净值对比（策略 vs 基准）
tab1, tab2, tab3, tab4 = st.tabs(["📊 净值与回撤", "🧾 交易明细", "📋 绩效分段表", "⬇️ 数据导出"])

with tab1:
    st.caption(("归一化净值（期初=1）。样本外起点：%s。" % oos_date) if oos_ts
               else "归一化净值（期初=1）。")
    comp = pd.DataFrame({
        "策略净值": nav / nav.iloc[0],
        bench_label: bench_nav / bench_nav.iloc[0],
    })
    st.line_chart(comp, height=320)
    st.caption("策略回撤曲线（净值相对历史新高的跌幅）。")
    dd = (nav / nav.cummax() - 1)
    st.area_chart(dd, height=220)

with tab2:
    trades = res["trades"].copy()
    show = trades[["cash", "nav", "traded", "cost", "n_pos"]].round(2)
    show.columns = ["现金(元)", "净值(元)", "当日成交额(元)", "当日费用(元)", "持仓数"]
    active = trades["traded"] > 0
    st.dataframe(show[active], use_container_width=True, height=380)
    st.caption(f"共 {int(active.sum())} 个交易日发生交易（仅列出有成交的日期），"
               f"总成交额 {res['total_traded']:,.0f} 元。")

with tab3:
    rows = []
    for seg_name, s in seg_stats(nav, oos_ts).items():
        rows.append({"区间": seg_name, "累计收益率": fmt_pct(s.get("累计收益率")),
                     "年化收益率": fmt_pct(s.get("年化收益率")),
                     "年化波动率": fmt_pct(s.get("年化波动率")),
                     "夏普比率": fmt_num(s.get("夏普比率")),
                     "最大回撤": fmt_pct(s.get("最大回撤")),
                     "Calmar比率": fmt_num(s.get("Calmar比率")),
                     "日胜率": fmt_pct(s.get("日胜率"))})
    bench_stats = perf.segment_stats(bench_nav)
    rows.append({"区间": f"基准：{bench_label}(全样本)", "累计收益率": fmt_pct(bench_stats.get("累计收益率")),
                 "年化收益率": fmt_pct(bench_stats.get("年化收益率")),
                 "年化波动率": fmt_pct(bench_stats.get("年化波动率")),
                 "夏普比率": fmt_num(bench_stats.get("夏普比率")),
                 "最大回撤": fmt_pct(bench_stats.get("最大回撤")),
                 "Calmar比率": fmt_num(bench_stats.get("Calmar比率")),
                 "日胜率": fmt_pct(bench_stats.get("日胜率"))})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption(f"当前设置：{title_asset}｜{strategy}｜无风险利率取0，一年按252个交易日计。")

with tab4:
    nav_df = pd.DataFrame({"策略净值": nav, bench_label: bench_nav})
    st.download_button("下载净值序列 (CSV)", nav_df.to_csv(encoding="utf-8-sig").encode("utf-8"),
                       file_name="backtest_nav.csv", mime="text/csv")
    tr = res["trades"][["cash", "nav", "traded", "cost", "n_pos"]].round(2)
    st.download_button("下载逐日台账 (CSV)", tr.to_csv(encoding="utf-8-sig").encode("utf-8"),
                       file_name="backtest_trades.csv", mime="text/csv")

    report = [
        f"回测报告  {dt.datetime.now():%Y-%m-%d %H:%M}",
        f"标的：{title_asset}    策略：{strategy}    参数：{params}",
        f"区间：{res['actual_start'].date()} ~ {res['actual_end'].date()}（{res['n_days']}个交易日）"
        f"    样本外起点：{oos_date}",
        f"成本：佣金{commission * 10000:.1f}‱ / 印花税{stamp * 10000:.1f}‱ / 滑点{slippage * 10000:.1f}‱    初始资金：{cash0:,.0f}元",
        "",
        f"全样本：年化 {fmt_pct(stats_all.get('年化收益率'))}，夏普 {fmt_num(stats_all.get('夏普比率'))}，"
        f"最大回撤 {fmt_pct(stats_all.get('最大回撤'))}，换手 {res['annual_turnover']:.1f} 倍/年",
        f"总交易成本：{res['total_cost']:,.0f} 元（总成交额 {res['total_traded']:,.0f} 元）",
    ]
    if stats_oos:
        report.append(f"样本外：年化 {fmt_pct(stats_oos.get('年化收益率'))}，夏普 {fmt_num(stats_oos.get('夏普比率'))}，"
                      f"最大回撤 {fmt_pct(stats_oos.get('最大回撤'))}")
    st.download_button("下载回测报告 (TXT)", "\n".join(report).encode("utf-8"),
                       file_name="backtest_report.txt", mime="text/plain")
