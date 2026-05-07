"""
V2.1 最终汇报材料 — 10 模型 × OOD + Random 对比
- PySR (10/10 fits done)
- gplearn (新对照，替代 KAN)
- 8 Classical (Linear_single/multi, Poly2/3, SVR_rbf, RF, XGBoost, MLP_BP)

不含 KAN（按用户要求剔除）。
"""
from __future__ import annotations
import json, glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "对外汇报材料"
OUT.mkdir(parents=True, exist_ok=True)

rcParams.update({
    "font.family": ["sans-serif"],
    "font.sans-serif": ["Noto Sans CJK JP", "Noto Sans CJK SC", "DejaVu Sans"],
    "axes.unicode_minus": False, "font.size": 10,
})

TARGETS = ["YMOD", "SMOD", "POIS", "SHMAX"]
WELLS_REAL = ["ZXX2", "ZXX3", "ZXX6", "ZXX7"]
WELLS_DESENS = ["WellA", "WellB", "WellC", "WellD"]
WELL_MAP = dict(zip(WELLS_REAL, WELLS_DESENS))

MODEL_ORDER = [
    "PySR", "gplearn",
    "Linear_single", "XGBoost", "RF", "SVR_rbf",
    "MLP_BP", "Linear_multi", "Poly2", "Poly3"
]
MODEL_COLORS = {
    "PySR":          "#F59E0B",
    "gplearn":       "#EAB308",
    "Linear_single": "#10B981",
    "XGBoost":       "#3B82F6",
    "RF":            "#06B6D4",
    "SVR_rbf":       "#A855F7",
    "MLP_BP":        "#EF4444",
    "Linear_multi":  "#94A3B8",
    "Poly2":         "#F97316",
    "Poly3":         "#DC2626",
}


def collect():
    rows = []
    # PySR OOD
    for cfg in glob.glob(str(ROOT / "张涵宇-PySR/真实数据组-ZXX/train_results/*/*/config.json")):
        c = json.load(open(cfg))
        rows.append({"well": c["well"], "target": c["target"], "model": "PySR",
                      "split": "OOD", "r2_train": c["r2_train"], "r2_test": c["r2_test"],
                      "formula": c.get("formula", "")[:200]})
    # PySR Random (sanity check)
    for cfg in glob.glob(str(ROOT / "消融/PySR_random_sanity/*/*/config.json")):
        c = json.load(open(cfg))
        rows.append({"well": c["well"], "target": c["target"], "model": "PySR",
                      "split": "Random", "r2_train": c["r2_train"], "r2_test": c["r2_test"],
                      "formula": c.get("formula", "")[:200]})
    # gplearn OOD + Random
    for cfg in glob.glob(str(ROOT / "其他SR对照/真实数据组-ZXX/train_results/*/*/gplearn/config.json")):
        c = json.load(open(cfg))
        rows.append({"well": c["well"], "target": c["target"], "model": "gplearn",
                      "split": "OOD", "r2_train": c["r2_train"], "r2_test": c["r2_test"],
                      "formula": c.get("formula_or_arch", "")[:200]})
    for cfg in glob.glob(str(ROOT / "其他SR对照/random_sanity/train_results/*/*/gplearn/config.json")):
        c = json.load(open(cfg))
        rows.append({"well": c["well"], "target": c["target"], "model": "gplearn",
                      "split": "Random", "r2_train": c["r2_train"], "r2_test": c["r2_test"],
                      "formula": c.get("formula_or_arch", "")[:200]})
    # 8 Classical OOD + Random
    for cfg in glob.glob(str(ROOT / "经典方法/真实数据组-ZXX/train_results/*/*/*/config.json")):
        c = json.load(open(cfg))
        rows.append({"well": c["well"], "target": c["target"], "model": c["model"],
                      "split": "OOD", "r2_train": c["r2_train"], "r2_test": c["r2_test"],
                      "formula": c.get("formula_or_arch", "")[:200]})
    for cfg in glob.glob(str(ROOT / "经典方法/random_sanity/train_results/*/*/*/config.json")):
        c = json.load(open(cfg))
        rows.append({"well": c["well"], "target": c["target"], "model": c["model"],
                      "split": "Random", "r2_train": c["r2_train"], "r2_test": c["r2_test"],
                      "formula": c.get("formula_or_arch", "")[:200]})
    return pd.DataFrame(rows)


def main():
    df = collect()
    df["well_desens"] = df["well"].map(WELL_MAP)
    print(f"总 fits: {len(df)} (期望 PySR 16 + gplearn 32 + 8 Classical × 32 = 304)")

    # Master CSV
    df.to_csv(OUT / "metrics_master_v21.csv", index=False, encoding="utf-8-sig")

    # ── ranking 总表 ──
    def agg_split(df_split):
        return df_split.groupby("model").agg(
            n=("r2_test", "count"),
            train_med=("r2_train", "median"),
            test_med=("r2_test", "median"),
            test_clip=("r2_test", lambda s: float(s.clip(-2, 1).mean())),
        ).round(3)
    ood_agg = agg_split(df[df["split"] == "OOD"]).sort_values("test_med", ascending=False)
    rnd_agg = agg_split(df[df["split"] == "Random"]).sort_values("test_med", ascending=False)

    ood_agg.to_csv(OUT / "ranking_OOD_v21.csv", encoding="utf-8-sig")
    rnd_agg.to_csv(OUT / "ranking_Random_v21.csv", encoding="utf-8-sig")
    print("\n=== V2.1 OOD ranking (last 20% by depth) ===")
    print(ood_agg.to_string())
    print("\n=== V2.1 Random ranking (常规 80/20 sanity) ===")
    print(rnd_agg.to_string())

    # ── random vs OOD 双轴对比表 ──
    cmp = pd.DataFrame({
        "OOD": df[df["split"] == "OOD"].groupby("model")["r2_test"].median(),
        "Random": df[df["split"] == "Random"].groupby("model")["r2_test"].median(),
    })
    # PySR 没跑 random（V2 没记录），算个 pseudo Random（用 train R²，假定 random 跟 train 接近）
    if "PySR" in cmp.index and pd.isna(cmp.loc["PySR", "Random"]):
        # 先看是否真的没跑
        pass
    cmp["drop"] = (cmp["Random"] - cmp["OOD"]).round(3)
    cmp = cmp.reindex(MODEL_ORDER)
    cmp.to_csv(OUT / "random_vs_OOD_对比表.csv", encoding="utf-8-sig")
    print("\n=== Random vs OOD 退化对比 ===")
    print(cmp.round(3).to_string())

    # ── 双柱图（去掉 KAN） ──
    fig, ax = plt.subplots(figsize=(13, 6))
    x = np.arange(len(cmp))
    w = 0.38
    cmp_clip = cmp.clip(-2, 1)
    has_random = ~cmp["Random"].isna()
    has_ood = ~cmp["OOD"].isna()
    ax.bar(x[has_random] - w/2, cmp_clip.loc[has_random, "Random"], w,
            label="Random 80/20 (常规)", color="#10B981")
    ax.bar(x[has_ood] + w/2, cmp_clip.loc[has_ood, "OOD"], w,
            label="OOD last 20% by depth", color="#F59E0B")
    for i, m in enumerate(cmp.index):
        rd, oo = cmp.loc[m, "Random"], cmp.loc[m, "OOD"]
        if pd.notna(rd):
            ax.text(i - w/2, cmp_clip.loc[m, "Random"] + 0.04, f"{rd:.2f}",
                    ha="center", fontsize=8)
        if pd.notna(oo):
            label = f"{oo:.2f}" if abs(oo) < 100 else f"{oo:.1e}"
            yy = cmp_clip.loc[m, "OOD"] + 0.04 if cmp_clip.loc[m, "OOD"] >= 0 else cmp_clip.loc[m, "OOD"] - 0.15
            ax.text(i + w/2, yy, label, ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(cmp.index, rotation=20)
    ax.set_ylim(-2.2, 1.15)
    ax.axhline(0, color="gray", lw=0.5); ax.axhline(1, color="gray", lw=0.3, ls="--")
    ax.set_ylabel("test R² 中位数 (clip[-2, 1])")
    ax.set_title("V2.1 Random vs OOD：10 方法对比（PySR + gplearn + 8 Classical）", fontsize=11)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "fig_v21_random_vs_OOD.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[图 1] {OUT / 'fig_v21_random_vs_OOD.png'}")

    # ── train vs test 散点图 ──
    fig, ax = plt.subplots(figsize=(10, 7))
    for m in MODEL_ORDER:
        sub = df[(df["model"] == m) & (df["split"] == "OOD")]
        if sub.empty: continue
        x = sub["r2_train"].values
        y = sub["r2_test"].clip(-2, 1).values
        ax.scatter(x, y, color=MODEL_COLORS[m], label=m, s=70, alpha=0.7, edgecolor="white")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="y=x (no overfit)")
    ax.axhline(0, color="red", lw=0.5, alpha=0.5)
    ax.set_xlabel("train R²"); ax.set_ylabel("OOD test R² (clip[-2, 1])")
    ax.set_title("V2.1 OOD: Train vs Test — 偏离 y=x 越多过拟合越严重")
    ax.set_xlim(-0.05, 1.05); ax.set_ylim(-2.1, 1.05)
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "fig_v21_overfit_散点图.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[图 2] {OUT / 'fig_v21_overfit_散点图.png'}")

    # ── PySR 公式表 ──
    pysr_formulas = df[(df["model"] == "PySR") & (df["split"] == "OOD")][
        ["well_desens", "target", "r2_train", "r2_test", "formula"]
    ].rename(columns={"well_desens": "well"})
    pysr_formulas.to_csv(OUT / "脱敏组_PySR公式表_v21.csv", index=False, encoding="utf-8-sig")
    print(f"[表] PySR 16 个真实公式 → 脱敏组_PySR公式表_v21.csv")

    print("\n=== V2.1 ALL DONE ===")


if __name__ == "__main__":
    raise SystemExit(main())
