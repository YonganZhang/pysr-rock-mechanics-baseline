"""
异常值诊断 — 画 3 张图肉眼看分布异常
1. Boxplot 矩阵：4 井 × 13 变量，一眼看 outliers
2. Histogram：每个变量分布形状（线性 + log scale 双视图）
3. Depth scatter：每个目标 vs 深度，看异常聚集深度

输出到 真跑结果-baseline-v2/_diagnostics/
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns

# 复用张涵宇数据加载
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "学生-张涵宇-PYSR改进/.trae/skills/pysr_equation_fit"))
from a import build_dataset_from_dir

OUT = Path(__file__).resolve().parents[1] / "_diagnostics"
OUT.mkdir(parents=True, exist_ok=True)

rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "axes.unicode_minus": False, "font.size": 8,
    "axes.titlesize": 9, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
})

WHITELIST = ["DTC","DTS","DEN","GR","CNL","LLD","LLS","RMSC","CAL"]
TARGETS = ["YMOD","SMOD","POIS","SHMAX"]
ALL_VARS = TARGETS + WHITELIST  # 4 + 9 = 13
WMAP = {"ZXX2": "Well 1", "ZXX3": "Well 2", "ZXX6": "Well 3", "ZXX7": "Well 4"}


def load_data():
    df = build_dataset_from_dir(ROOT / "学生-张涵宇-PYSR改进/data", keep_common_columns_only=False)
    df.loc[df["WELLNAME"]=="ZXX3井","WELLNAME"]="ZXX3"
    df.loc[df["WELLNAME"]=="ZXX7井","WELLNAME"]="ZXX7"
    df.loc[df["WELLNAME"]=="足212（导眼井）","WELLNAME"]="ZXX2"
    return df


def fig_boxplot_matrix():
    """图 1：4 井 × 13 变量 boxplot 矩阵 — 一眼看 outliers"""
    df = load_data()
    fig, axes = plt.subplots(13, 4, figsize=(11, 22), sharex=False, sharey=False)
    for row, var in enumerate(ALL_VARS):
        for col, w in enumerate(["ZXX2","ZXX3","ZXX6","ZXX7"]):
            ax = axes[row, col]
            sub = df[df["WELLNAME"]==w]
            if var not in sub.columns:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
                ax.set_xticks([]); ax.set_yticks([])
                if col == 0: ax.set_ylabel(var, fontsize=10, fontweight="bold")
                if row == 0: ax.set_title(WMAP[w], fontsize=10, fontweight="bold")
                continue
            v = pd.to_numeric(sub[var], errors="coerce").dropna()
            v = v[v.abs() > 1e-9]  # 去 0
            n_total = len(v)
            n_neg = (v < 0).sum()
            n_extreme_neg = (v < -100).sum()
            n_extreme_pos = (v > v.quantile(0.99) * 10).sum() if v.quantile(0.99) > 0 else 0
            color = "#FF6B6B" if n_extreme_neg > 0 else "#4ECDC4"
            bp = ax.boxplot(v, vert=True, patch_artist=True,
                            boxprops=dict(facecolor=color, alpha=0.5),
                            medianprops=dict(color="black", linewidth=1.2),
                            flierprops=dict(marker="o", markersize=2, alpha=0.5,
                                            markerfacecolor="red", markeredgecolor="none"))
            # 标注：n / n_neg / n_extreme
            note = f"n={n_total}"
            if n_neg > 0: note += f"\n负值={n_neg}"
            if n_extreme_neg > 0: note += f"\n<-100: {n_extreme_neg}"
            ax.text(1.15, v.median(), note, fontsize=6, va="center")
            ax.set_xticks([])
            if col == 0: ax.set_ylabel(var, fontsize=10, fontweight="bold")
            if row == 0: ax.set_title(WMAP[w], fontsize=10, fontweight="bold")
    fig.suptitle("Outlier Diagnosis: Boxplot Matrix (Red box = contains <-100 outliers)",
                 fontsize=12, y=1.005)
    fig.tight_layout()
    fig.savefig(OUT / "diag1_boxplot_matrix.png", dpi=150)
    plt.close(fig)


def fig_histogram_grid():
    """图 2：每个变量的 histogram（4 井 overlay），看分布形状"""
    df = load_data()
    fig, axes = plt.subplots(4, 4, figsize=(13, 11))
    axes_flat = axes.flatten()
    colors = {"ZXX2": "#440154", "ZXX3": "#3b528b", "ZXX6": "#21918c", "ZXX7": "#fde725"}

    for i, var in enumerate(ALL_VARS):
        ax = axes_flat[i]
        for w in ["ZXX2","ZXX3","ZXX6","ZXX7"]:
            sub = df[df["WELLNAME"]==w]
            if var not in sub.columns: continue
            v = pd.to_numeric(sub[var], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            v = v[v.abs() > 1e-9]
            # 用 robust range（1%~99% percentile）画图，避免极端值撑爆
            if len(v) < 10: continue
            lo, hi = v.quantile(0.001), v.quantile(0.999)
            v_plot = v[(v >= lo) & (v <= hi)]
            if len(v_plot) < 10: continue
            ax.hist(v_plot, bins=80, alpha=0.4, color=colors[w], label=WMAP[w], edgecolor="none")
        ax.set_title(var, fontsize=10, fontweight="bold", loc="left")
        ax.set_yscale("log")
        ax.grid(linestyle="--", alpha=0.3)
        if i == 0: ax.legend(fontsize=7, loc="upper right")
    # 空 panels
    for i in range(len(ALL_VARS), len(axes_flat)):
        axes_flat[i].axis("off")
    fig.suptitle("Variable Distributions (log y-scale, 4 wells overlay)",
                 fontsize=12, y=1.0)
    fig.tight_layout()
    fig.savefig(OUT / "diag2_histograms.png", dpi=150)
    plt.close(fig)


def fig_depth_scatter():
    """图 3: 4 目标 vs depth 散点（4 井 1 行 4 列），看异常聚集深度"""
    df = load_data()
    fig, axes = plt.subplots(4, 4, figsize=(14, 12))
    for ti, tgt in enumerate(TARGETS):
        for wi, w in enumerate(["ZXX2","ZXX3","ZXX6","ZXX7"]):
            ax = axes[ti, wi]
            sub = df[df["WELLNAME"]==w][["DEPTH", tgt]].apply(pd.to_numeric, errors="coerce").dropna()
            sub = sub[sub[tgt].abs() > 1e-9]
            if len(sub) < 10:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
                continue
            # 标记异常：< 0 红色，>0 但 > q99*5 也红色
            q99 = sub[tgt].quantile(0.99)
            q01 = sub[tgt].quantile(0.01)
            normal_mask = (sub[tgt] > 0) & (sub[tgt] < q99 * 5) & (sub[tgt] > q01 * 0.1 if q01 > 0 else True)
            ax.scatter(sub.loc[normal_mask, tgt], sub.loc[normal_mask, "DEPTH"],
                        s=2, c="#21918c", alpha=0.5, label="normal" if (ti==0 and wi==0) else None)
            ax.scatter(sub.loc[~normal_mask, tgt], sub.loc[~normal_mask, "DEPTH"],
                        s=8, c="red", alpha=0.8, marker="x",
                        label="outlier" if (ti==0 and wi==0) else None)
            ax.invert_yaxis()
            ax.set_title(f"{WMAP[w]}/{tgt}", fontsize=9, loc="left")
            n_out = (~normal_mask).sum()
            ax.text(0.98, 0.02, f"outlier={n_out}", ha="right", va="bottom",
                    transform=ax.transAxes, fontsize=8, color="red",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="red", alpha=0.8))
            if wi == 0: ax.set_ylabel("Depth (m)", fontsize=8)
            if ti == 3: ax.set_xlabel(tgt, fontsize=8)
            if ti == 0 and wi == 0: ax.legend(fontsize=7, loc="upper right")
    fig.suptitle("Target Values vs Depth (red ✕ = outliers; q99×5 or <0 cutoff)",
                 fontsize=12, y=1.0)
    fig.tight_layout()
    fig.savefig(OUT / "diag3_depth_scatter.png", dpi=150)
    plt.close(fig)


def report_outlier_stats():
    """文字报告：每变量每井异常值统计"""
    df = load_data()
    rows = []
    for w in ["ZXX2","ZXX3","ZXX6","ZXX7"]:
        sub = df[df["WELLNAME"]==w]
        for var in ALL_VARS:
            if var not in sub.columns: continue
            v = pd.to_numeric(sub[var], errors="coerce").dropna()
            v = v[v.abs() > 1e-9]
            if len(v) < 10: continue
            rows.append({
                "well": WMAP[w], "var": var,
                "n": len(v),
                "min": v.min(),
                "q01": v.quantile(0.01),
                "median": v.median(),
                "q99": v.quantile(0.99),
                "max": v.max(),
                "n_neg": (v < 0).sum(),
                "n_lt_-1": (v < -1).sum(),
                "n_lt_-100": (v < -100).sum(),
                "n_lt_-500": (v < -500).sum(),
                "n_lt_-9000": (v < -9000).sum(),
            })
    df_stats = pd.DataFrame(rows)
    df_stats.to_csv(OUT / "outlier_stats.csv", index=False, encoding="utf-8-sig")
    return df_stats


def main():
    print("=== 1/4 boxplot matrix (4 井 × 13 变量) ===")
    fig_boxplot_matrix()
    print("=== 2/4 histograms (4 井 overlay) ===")
    fig_histogram_grid()
    print("=== 3/4 depth scatter (异常深度位置) ===")
    fig_depth_scatter()
    print("=== 4/4 文字统计 ===")
    stats = report_outlier_stats()
    # 关键摘要
    print("\n=== 异常值汇总 (前 20 条 n_neg > 0) ===")
    crit = stats[stats["n_neg"] > 0].sort_values("n_neg", ascending=False)
    print(crit.head(30).to_string(index=False))
    print(f"\n所有诊断 → {OUT}")


if __name__ == "__main__":
    main()
