# -*- coding: utf-8 -*-
"""绩效评估与可视化模块"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from indicators import sma, bollinger

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

TRADING_DAYS = 252


def max_drawdown(nav: pd.Series) -> float:
    return float((nav / nav.cummax() - 1).min())


def segment_stats(nav: pd.Series) -> dict:
    """对一段净值序列计算绩效指标（无风险利率取0）"""
    nav = nav.dropna()
    if len(nav) < 2:
        return {}
    ret = nav.pct_change().dropna()
    total = float(nav.iloc[-1] / nav.iloc[0] - 1)
    ann = float((1 + total) ** (TRADING_DAYS / len(ret)) - 1)
    vol = float(ret.std(ddof=0) * np.sqrt(TRADING_DAYS))
    sharpe = float(ret.mean() / ret.std(ddof=0) * np.sqrt(TRADING_DAYS)) if ret.std() > 0 else np.nan
    mdd = max_drawdown(nav)
    calmar = ann / abs(mdd) if mdd < 0 else np.nan
    return {"累计收益率": total, "年化收益率": ann, "年化波动率": vol,
            "夏普比率": sharpe, "最大回撤": mdd, "Calmar比率": calmar,
            "日胜率": float((ret > 0).mean())}


def report(res: dict) -> dict:
    """汇总引擎输出为指标字典"""
    stats = segment_stats(res["nav"])
    stats["年化换手率"] = res["annual_turnover"]
    stats["总交易成本"] = res["total_cost"]
    return stats


def stats_table(results: dict[str, pd.Series]) -> pd.DataFrame:
    """results: {策略名: nav序列} -> 指标对比表"""
    rows = {name: report({"nav": nav, "annual_turnover": np.nan,
                          "total_cost": np.nan}) for name, nav in results.items()}
    return pd.DataFrame(rows).T


def plot_nav(results: dict[str, pd.Series], title: str, path: str,
             oos_start: str | None = None, figsize=(9, 5.2)):
    fig, ax = plt.subplots(figsize=figsize)
    for name, nav in results.items():
        nav = nav.dropna()
        nav = nav / nav.iloc[0]                    # 归一化：期初=1
        ax.plot(nav.index, nav.values, label=name, lw=1.4)
    if oos_start:
        ax.axvline(pd.Timestamp(oos_start), color="gray", ls="--", lw=1)
        ax.text(pd.Timestamp(oos_start), ax.get_ylim()[1] * 0.98,
                " 样本外起点", va="top", fontsize=9, color="gray")
    ax.set_title(title, fontsize=13)
    ax.set_ylabel("净值（期初=1）")
    ax.legend(frameon=False, fontsize=10, loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_drawdown(results: dict[str, pd.Series], title: str, path: str,
                  figsize=(9, 4.6)):
    fig, ax = plt.subplots(figsize=figsize)
    for name, nav in results.items():
        nav = nav.dropna()
        nav = nav / nav.iloc[0]
        dd = nav / nav.cummax() - 1
        ax.plot(dd.index, dd.values, label=name, lw=1.2)
    ax.set_title(title, fontsize=13)
    ax.set_ylabel("回撤")
    ax.legend(frameon=False, fontsize=10, loc="lower left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_bars(df: pd.DataFrame, title: str, path: str, ylabel: str,
              figsize=(8.6, 4.8)):
    """分组柱状图：df.index=策略, df.columns=成本情景"""
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(df.index))
    ncol = len(df.columns)
    width = 0.8 / max(ncol, 1)
    for j, colname in enumerate(df.columns):
        ax.bar(x + (j - (ncol - 1) / 2) * width, df[colname].values,
               width, label=str(colname))
    ax.set_xticks(x)
    ax.set_xticklabels(df.index, fontsize=10)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=13)
    ax.legend(frameon=False, fontsize=10)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_architecture(path: str):
    """系统四层架构图"""
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.axis("off")
    layers = [
        ("数据层", "akshare/新浪接口 → 后复权日线 → 本地CSV缓存 → 指标计算"),
        ("引擎层", "T日收盘信号 → T+1开盘撮合 → 佣金/印花税/滑点 → 现金持仓台账"),
        ("策略层", "双均线 · 布林带均值回归 · 横截面动量 · 买入持有基准"),
        ("评估层", "年化收益/夏普/最大回撤/Calmar/换手 → 净值与回撤图 → 对比实验"),
    ]
    colors = ["#dbe9f6", "#e8f0dc", "#f6ecdb", "#eadff0"]
    top, bottom = 0.96, 0.04
    n = len(layers)
    box_h = 0.15
    gap = (top - bottom - n * box_h) / (n - 1)
    for i, (name, desc) in enumerate(layers):
        y = top - box_h - i * (box_h + gap)      # 每层框的底边y
        ax.add_patch(plt.Rectangle((0.05, y), 0.15, box_h,
                                   fc=colors[i], ec="#555", lw=1))
        ax.text(0.125, y + box_h / 2, name, ha="center", va="center",
                fontsize=13, weight="bold")
        ax.add_patch(plt.Rectangle((0.28, y), 0.67, box_h,
                                   fc="white", ec="#555", lw=1))
        ax.text(0.615, y + box_h / 2, desc, ha="center", va="center", fontsize=10.5)
        if i < n - 1:
            ax.annotate("", xy=(0.615, y - gap + 0.004), xytext=(0.615, y - 0.004),
                        arrowprops=dict(arrowstyle="-|>", color="#444", lw=1.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_price_with_indicators(close: pd.Series, path: str, title: str):
    """标的走势与指标（图示数据模块输出）"""
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.plot(close.index, close.values, lw=1.1, label="收盘价（后复权）")
    ax.plot(sma(close, 5).index, sma(close, 5).values, lw=0.9, label="MA5")
    ax.plot(sma(close, 20).index, sma(close, 20).values, lw=0.9, label="MA20")
    ma, lo, hi = bollinger(close, 20, 2)
    ax.fill_between(ma.index, lo.values, hi.values, alpha=0.12, label="布林带(20,2)")
    ax.set_title(title, fontsize=13)
    ax.set_ylabel("价格（元，后复权）")
    ax.legend(frameon=False, fontsize=9, ncol=4, loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
