"""
出版级 9 张论文图（应用 Gemini 风格建议）

升级点（vs v1）：
- viridis/coolwarm 配色方案，色盲友好
- Helvetica/Arial 字体，统一字号体系（suptitle 12, title 10, label 9）
- DPI 300+
- 移除顶部+右侧 spines
- 全局 fig.legend 替代子图独立图例
- 散点图 alpha + y=x 参考线 + aspect equal
- depth profile 用 axvspan 分区训练/测试段
"""
from __future__ import annotations
import json, glob
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns

# 出版级全局风格（IEEE/Nature 期刊）
rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Noto Sans CJK JP", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.titlesize": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "lines.linewidth": 1.0,
})

ROOT = Path(__file__).resolve().parents[3] / "真跑结果-baseline-v2"
OUT = Path(__file__).resolve().parents[1] / "figures"
OUT.mkdir(parents=True, exist_ok=True)

WMAP = {"ZXX2": "Well 1", "ZXX3": "Well 2", "ZXX6": "Well 3", "ZXX7": "Well 4"}
TARGETS = ["YMOD", "SMOD", "POIS", "SHMAX"]
WELLS_REAL = ["ZXX2", "ZXX3", "ZXX6", "ZXX7"]
WELLS_DESENS = ["Well 1", "Well 2", "Well 3", "Well 4"]

# Viridis-derived 4 色（色盲友好）
COLORS = {
    "GWMF-SR":  "#440154",  # viridis dark purple
    "XGBoost":  "#3b528b",  # viridis blue
    "RF":       "#21918c",  # viridis teal
    "Poly3":    "#fde725",  # viridis yellow
}


def collect_pysr():
    rows = []
    for cfg in glob.glob(str(ROOT / "张涵宇-PySR/真实数据组-ZXX/train_results/*/*/config.json")):
        c = json.load(open(cfg))
        rows.append({"well": c["well"], "target": c["target"],
                      "r2_train": c["r2_train"], "r2_test": c["r2_test"],
                      "formula": c.get("formula", "")[:200]})
    return pd.DataFrame(rows)


def collect_classical():
    rows = []
    for cfg in glob.glob(str(ROOT / "经典方法/真实数据组-ZXX/train_results/*/*/*/config.json")):
        c = json.load(open(cfg))
        rows.append({"well": c["well"], "target": c["target"], "model": c["model"],
                      "r2_train": c["r2_train"], "r2_test": c["r2_test"]})
    return pd.DataFrame(rows)


def collect_ablation_A():
    rows = []
    for cfg in glob.glob(str(ROOT / "消融/A_特征消融/*/*/*/*/config.json")):
        c = json.load(open(cfg))
        rows.append({"well": c["well"], "target": c["target"], "model": c["model"],
                      "removed": c["removed_feature"],
                      "r2_test": c["r2_test"], "r2_train": c["r2_train"]})
    return pd.DataFrame(rows)


def fig_r2_comparison():
    pysr = collect_pysr(); cls = collect_classical()
    pysr["model"] = "GWMF-SR"
    df = pd.concat([pysr[["well","target","model","r2_test"]],
                     cls[["well","target","model","r2_test"]]])
    df["well_disp"] = df["well"].map(WMAP)
    keep_models = ["GWMF-SR", "XGBoost", "RF", "Poly3"]

    fig, axes = plt.subplots(1, 4, figsize=(7.5, 2.6), sharey=True)
    panel_labels = ["(a) YMOD", "(b) SMOD", "(c) POIS", "(d) SHMAX"]
    for i, tgt in enumerate(TARGETS):
        ax = axes[i]
        sub = df[df["target"] == tgt]
        x = np.arange(len(WELLS_DESENS))
        w = 0.20
        for j, m in enumerate(keep_models):
            ms = sub[sub["model"] == m].set_index("well_disp").reindex(WELLS_DESENS)
            vals = ms["r2_test"].clip(-1, 1).values
            ax.bar(x + (j-1.5)*w, vals, w, color=COLORS[m],
                    edgecolor="white", linewidth=0.3,
                    label=m if i == 0 else None)
        ax.set_xticks(x); ax.set_xticklabels(WELLS_DESENS, fontsize=8)
        ax.set_title(panel_labels[i], fontsize=10, loc="left")
        ax.set_ylim(-1.05, 1.05)
        ax.axhline(0, color="black", lw=0.4)
        ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
        if i == 0: ax.set_ylabel("OOD test $R^2$")
    fig.legend(loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02),
                frameon=False, fontsize=9)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(OUT / "fig_r2_comparison.png", dpi=300)
    plt.close(fig)


def fig_r2_improvement():
    pysr = collect_pysr(); cls = collect_classical()
    xgb = cls[cls["model"] == "XGBoost"].set_index(["well","target"])["r2_test"]
    p = pysr.set_index(["well","target"])["r2_test"]
    delta = (p - xgb).clip(-1.5, 1.5).unstack("target")[TARGETS]
    delta.index = [WMAP[w] for w in delta.index]

    fig, ax = plt.subplots(figsize=(4.5, 3))
    sns.heatmap(delta, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                cbar_kws={"label": "$\\Delta R^2$ (GWMF-SR − XGBoost)"},
                ax=ax, vmin=-1, vmax=1,
                linewidths=0.5, linecolor="white",
                annot_kws={"fontsize": 9})
    ax.set_title("R² Improvement Across Wells & Targets", fontsize=10)
    ax.set_xlabel(""); ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(OUT / "fig_r2_improvement_heatmap.png", dpi=300)
    plt.close(fig)


def fig_scatter():
    methods = {
        "GWMF-SR":  (ROOT / "张涵宇-PySR/真实数据组-ZXX/train_results", COLORS["GWMF-SR"]),
        "XGBoost":  (ROOT / "经典方法/真实数据组-ZXX/train_results", COLORS["XGBoost"]),
        "RF":       (ROOT / "经典方法/真实数据组-ZXX/train_results", COLORS["RF"]),
        "Poly3":    (ROOT / "经典方法/真实数据组-ZXX/train_results", COLORS["Poly3"]),
    }
    fig, axes = plt.subplots(2, 2, figsize=(7, 7))
    axes = axes.ravel()
    for i, m in enumerate(methods):
        ax = axes[i]
        root, color = methods[m]
        all_meas, all_pred = [], []
        for w in WELLS_REAL:
            for tgt in TARGETS:
                if m == "GWMF-SR":
                    p = root / w / tgt / "predictions.csv"
                else:
                    p = root / w / tgt / m / "predictions.csv"
                if not p.exists(): continue
                d = pd.read_csv(p)
                d = d[(d["measured"].abs() > 1e-9) & (d["split"] == "test")]
                if len(d) == 0: continue
                m_min, m_max = d["measured"].min(), d["measured"].max()
                rng = m_max - m_min
                d_clip = d[(d["predicted"] >= m_min - rng) &
                            (d["predicted"] <= m_max + rng)]
                # 归一化到 [0, 1]
                if rng > 0:
                    me = (d_clip["measured"] - m_min) / rng
                    pr = (d_clip["predicted"] - m_min) / rng
                    all_meas.extend(me.values)
                    all_pred.extend(pr.values)
        if all_meas:
            ax.scatter(all_meas, all_pred, s=8, alpha=0.4, color=color,
                        edgecolor="none")
            from numpy import corrcoef
            r2 = corrcoef(all_meas, all_pred)[0,1]**2
            ax.text(0.05, 0.92, f"$R^2$ = {r2:.3f}", transform=ax.transAxes,
                    fontsize=9, bbox=dict(boxstyle="round", facecolor="white",
                                            edgecolor="gray", alpha=0.8))
        ax.plot([-0.2, 1.2], [-0.2, 1.2], "k--", lw=0.7, alpha=0.5)
        ax.set_xlim(-0.1, 1.1); ax.set_ylim(-0.1, 1.1)
        ax.set_aspect("equal", "box")
        ax.set_title(f"({chr(97+i)}) {m}", fontsize=10, loc="left")
        if i % 2 == 0: ax.set_ylabel("Predicted (normalized)")
        if i >= 2: ax.set_xlabel("Measured (normalized)")
        ax.grid(linestyle="--", alpha=0.3)
    fig.suptitle("OOD Test: Predicted vs Measured (4 wells × 4 targets pooled)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "fig_scatter_pred_vs_true.png", dpi=300)
    plt.close(fig)


def fig_feature_heatmap():
    A = collect_ablation_A()
    A = A[A["r2_test"].notna()].copy()
    piv = A.groupby(["model", "removed"])["r2_test"].median().unstack("removed")
    features = ["DTC","DTS","DEN","GR","CNL","LLD","LLS","RMSC","CAL"]
    piv = piv[["(none)"] + features]
    delta = pd.DataFrame({f: (piv["(none)"] - piv[f]).clip(0, 1.5) for f in features})

    fig, ax = plt.subplots(figsize=(7, 3.2))
    sns.heatmap(delta, annot=True, fmt=".2f", cmap="viridis",
                cbar_kws={"label": "$R^2$ Drop (Removed − Baseline)"},
                ax=ax, vmin=0, vmax=1.0,
                linewidths=0.5, linecolor="white",
                annot_kws={"fontsize": 8})
    ax.set_title("Leave-One-Feature-Out Ablation: OOD R² Drop", fontsize=10)
    ax.set_xlabel("Removed Feature (Darker = More Critical)"); ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(OUT / "fig_feature_heatmap.png", dpi=300)
    plt.close(fig)


def fig_xgb_importance():
    cfg = json.load(open(ROOT / "经典方法/真实数据组-ZXX/train_results/ZXX2/YMOD/XGBoost/config.json"))
    fi = cfg.get("feature_importance", {})
    if not fi: return
    fi_sorted = sorted(fi.items(), key=lambda x: x[1], reverse=True)
    feats, vals = zip(*fi_sorted)

    fig, ax = plt.subplots(figsize=(5, 3.5))
    bars = ax.barh(range(len(feats)), vals, color="#4682b4",
                    edgecolor="none")
    ax.set_yticks(range(len(feats))); ax.set_yticklabels(feats)
    ax.invert_yaxis()
    for i, v in enumerate(vals):
        ax.text(v + max(vals)*0.01, i, f"{v:.3f}", va="center", fontsize=8)
    ax.set_xlabel("Feature Importance (Gain)")
    ax.set_title("XGBoost Importance: Well 1 / YMOD", fontsize=10)
    ax.spines["left"].set_color("gray")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(OUT / "fig_xgb_importance_zxx2.png", dpi=300)
    plt.close(fig)


def fig_shap_beeswarm():
    rows = []
    for cfg in glob.glob(str(ROOT / "经典方法/真实数据组-ZXX/train_results/*/YMOD/RF/config.json")):
        c = json.load(open(cfg))
        fi = c.get("feature_importance", {})
        for f, v in fi.items():
            rows.append({"feature": f, "importance": v, "well": c["well"]})
    if not rows: return
    fi_df = pd.DataFrame(rows)
    order = fi_df.groupby("feature")["importance"].median().sort_values(ascending=False).index

    fig, ax = plt.subplots(figsize=(6, 3.5))
    sns.boxplot(data=fi_df, x="importance", y="feature", order=order,
                color="#7FAACE", linewidth=0.8, fliersize=0, ax=ax)
    sns.stripplot(data=fi_df, x="importance", y="feature", order=order,
                   color="#1f4e79", size=5, alpha=0.85, ax=ax,
                   edgecolor="white", linewidth=0.4)
    ax.set_xlabel("RF Feature Importance")
    ax.set_ylabel("")
    ax.set_title("Feature Importance Distribution (4 wells, YMOD)", fontsize=10)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(OUT / "fig_shap_beeswarm.png", dpi=300)
    plt.close(fig)


def fig_shap_dep():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "学生-张涵宇-PYSR改进/.trae/skills/pysr_equation_fit"))
    from a import build_dataset_from_dir
    df = build_dataset_from_dir(Path(__file__).resolve().parents[3] / "学生-张涵宇-PYSR改进/data", keep_common_columns_only=False)
    df.loc[df["WELLNAME"]=="ZXX3井","WELLNAME"] = "ZXX3"
    sub = df[df["WELLNAME"]=="ZXX3"][["DEPTH","DTS","DTC","YMOD"]].apply(pd.to_numeric, errors="coerce").dropna()
    sub = sub[sub["YMOD"].abs() > 1e-9]

    fig, ax = plt.subplots(figsize=(5, 3.5))
    sc = ax.scatter(sub["DTS"], sub["YMOD"], s=8, alpha=0.5,
                     c=sub["DTC"], cmap="coolwarm", edgecolor="none")
    cbar = plt.colorbar(sc, ax=ax, label="DTC (interaction)")
    ax.set_xlabel("DTS (Shear-wave Slowness)")
    ax.set_ylabel("YMOD (Young's Modulus)")
    ax.set_title("DTS-YMOD Dependence (Well 2, color=DTC)", fontsize=10)
    ax.grid(linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_shap_dep_dtc.png", dpi=300)
    plt.close(fig)


def fig_cross_validation():
    pysr = collect_pysr()
    WHITE = ["DTC","DTS","DEN","GR","CNL","LLD","LLS","RMSC","CAL"]
    rows = []
    for _, r in pysr.iterrows():
        formula = r["formula"]
        used = [v for v in WHITE if v in formula]
        rows.append({"well": r["well"], "target": r["target"], "n_pysr_vars": len(used)})
    df_p = pd.DataFrame(rows)
    df_p["well_disp"] = df_p["well"].map(WMAP)
    pivot = df_p.pivot(index="well_disp", columns="target",
                        values="n_pysr_vars").reindex(WELLS_DESENS)
    pivot = pivot[TARGETS]

    fig, ax = plt.subplots(figsize=(4.5, 3))
    sns.heatmap(pivot, annot=True, fmt="d", cmap="rocket_r",
                cbar_kws={"label": "Number of Variables in GWMF-SR Formula"},
                ax=ax, vmin=1, vmax=8,
                linewidths=0.5, linecolor="white",
                annot_kws={"fontsize": 10, "fontweight": "bold"})
    ax.set_title("Formula Complexity: Variable Count per (Well, Target)", fontsize=10)
    ax.set_xlabel(""); ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(OUT / "fig_cross_validation.png", dpi=300)
    plt.close(fig)


def fig_depth_profiles():
    for tgt in TARGETS:
        fig, axes = plt.subplots(1, 4, figsize=(8, 6), sharey=False)
        for i, w in enumerate(WELLS_REAL):
            ax = axes[i]
            p_pysr = ROOT / "张涵宇-PySR/真实数据组-ZXX/train_results" / w / tgt / "predictions.csv"
            if not p_pysr.exists():
                ax.set_title(f"{WMAP[w]} (no data)")
                continue
            d = pd.read_csv(p_pysr)
            d = d[d["measured"].abs() > 1e-9].sort_values("depth")
            m_min, m_max = d["measured"].min(), d["measured"].max()
            rng = m_max - m_min
            x_lim = (m_min - rng*0.4, m_max + rng*0.4)

            tr = d[d["split"] == "train"]; te = d[d["split"] == "test"]
            # 测试段背景高亮
            if len(te) > 0:
                test_d_min = te["depth"].min()
                test_d_max = te["depth"].max()
                ax.axhspan(test_d_min, test_d_max, alpha=0.12,
                            color="#fde725", zorder=0)
            ax.scatter(d["measured"], d["depth"], s=2, c="black",
                        alpha=0.5, label="Measured" if i == 0 else None)
            ax.plot(tr["predicted"], tr["depth"], color=COLORS["GWMF-SR"],
                     lw=0.7, alpha=0.6, label="GWMF-SR (train)" if i == 0 else None)
            ax.plot(te["predicted"], te["depth"], color=COLORS["GWMF-SR"],
                     lw=1.5, label="GWMF-SR (test)" if i == 0 else None)
            ax.set_xlim(*x_lim); ax.invert_yaxis()
            ax.set_title(f"({chr(97+i)}) {WMAP[w]}", fontsize=10, loc="left")
            ax.set_xlabel(tgt)
            if i == 0:
                ax.set_ylabel("Depth (m)")
            ax.grid(linestyle="--", alpha=0.3)
        fig.legend(loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02),
                    frameon=False, fontsize=9)
        idx_map = {"YMOD": "a", "SMOD": "b", "POIS": "c", "SHMAX": "d"}
        fig.suptitle(f"Depth Profile: {tgt} (yellow band = OOD test)", fontsize=11)
        fig.tight_layout(rect=[0, 0.04, 1, 1])
        fig.savefig(OUT / f"fig_depth_{idx_map[tgt]}_{tgt}.png", dpi=300)
        plt.close(fig)


def main():
    fig_r2_comparison(); print("✓ fig_r2_comparison")
    fig_r2_improvement(); print("✓ fig_r2_improvement_heatmap")
    fig_scatter(); print("✓ fig_scatter_pred_vs_true")
    fig_feature_heatmap(); print("✓ fig_feature_heatmap")
    fig_xgb_importance(); print("✓ fig_xgb_importance_zxx2")
    fig_shap_beeswarm(); print("✓ fig_shap_beeswarm")
    fig_shap_dep(); print("✓ fig_shap_dep_dtc")
    fig_cross_validation(); print("✓ fig_cross_validation")
    fig_depth_profiles(); print("✓ fig_depth_a-d")


if __name__ == "__main__":
    main()
