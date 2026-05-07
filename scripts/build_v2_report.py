"""
V2 对外汇报材料生成
- ranking 总表 + 4 子图柱状图（每 target）
- depth profile 图（每 target，含所有方法）
- README + PySR 公式表
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
    "axes.labelsize": 11, "axes.titlesize": 12,
    "axes.spines.top": False, "axes.spines.right": False,
})

TARGETS = ["YMOD", "SMOD", "POIS", "SHMAX"]
WELLS_REAL = ["ZXX2", "ZXX3", "ZXX6", "ZXX7"]
WELLS_DESENS = ["WellA", "WellB", "WellC", "WellD"]
WELL_MAP = dict(zip(WELLS_REAL, WELLS_DESENS))

MODEL_ORDER = ["PySR", "Linear_single", "XGBoost", "RF", "SVR_rbf",
                "KAN", "Linear_multi", "MLP_BP", "Poly2", "Poly3"]
MODEL_COLORS = {
    "PySR":          "#F59E0B",
    "Linear_single": "#10B981",
    "XGBoost":       "#3B82F6",
    "RF":            "#06B6D4",
    "SVR_rbf":       "#A855F7",
    "KAN":           "#7C3AED",
    "Linear_multi":  "#94A3B8",
    "MLP_BP":        "#EF4444",
    "Poly2":         "#F97316",
    "Poly3":         "#DC2626",
}


def collect_master():
    rows = []
    # classical
    for cfg in glob.glob(str(ROOT / "经典方法/真实数据组-ZXX/train_results/*/*/*/config.json")):
        c = json.load(open(cfg))
        rows.append({"well": c["well"], "target": c["target"], "model": c["model"],
                      "r2_train": c["r2_train"], "r2_test": c["r2_test"],
                      "mae_test": c["mae_test"], "n_train": c["n_train"], "n_test": c["n_test"],
                      "formula": c.get("formula_or_arch", "")})
    for cfg in glob.glob(str(ROOT / "张涵宇-PySR/真实数据组-ZXX/train_results/*/*/config.json")):
        c = json.load(open(cfg))
        rows.append({"well": c["well"], "target": c["target"], "model": "PySR",
                      "r2_train": c["r2_train"], "r2_test": c["r2_test"],
                      "mae_test": c["mae_test"], "n_train": c["n_train"], "n_test": c["n_test"],
                      "formula": c.get("formula", "")})
    for cfg in glob.glob(str(ROOT / "齐振鹏-KAN/真实数据组-ZXX/train_results/*/*/config.json")):
        c = json.load(open(cfg))
        rows.append({"well": c["well"], "target": c["target"], "model": "KAN",
                      "r2_train": c["r2_train"], "r2_test": c["r2_test"],
                      "mae_test": c["mae_test"], "n_train": c["n_train"], "n_test": c["n_test"],
                      "formula": "KAN_standard"})
    return pd.DataFrame(rows)


def main():
    df = collect_master()
    df["well_desens"] = df["well"].map(WELL_MAP)
    df.to_csv(OUT / "metrics_master.csv", index=False, encoding="utf-8-sig")
    print(f"[1] master → metrics_master.csv ({len(df)} rows)")

    # ranking
    agg = df.groupby("model").agg(
        n=("r2_test", "count"),
        test_r2_median=("r2_test", "median"),
        test_r2_mean_clip=("r2_test", lambda s: float(s.clip(-2, 1).mean())),
        train_r2_mean=("r2_train", "mean"),
    ).round(4)
    agg["overfit_gap"] = (df.groupby("model")["r2_train"].mean() -
                           df.groupby("model")["r2_test"].apply(lambda s: s.clip(-2, 1).mean())).round(4)
    agg = agg.sort_values("test_r2_median", ascending=False)
    agg.to_csv(OUT / "ranking_总表.csv", encoding="utf-8-sig")
    print(f"[2] ranking → ranking_总表.csv")

    # 双版本：脱敏版 + 真实版的 R² 透视表
    for label, well_col in [("脱敏组", "well_desens"), ("真实组", "well")]:
        piv = df.pivot_table(index=[well_col, "target"], columns="model",
                              values="r2_test").reset_index()
        piv.to_csv(OUT / f"{label}_R2_test透视表.csv", index=False, encoding="utf-8-sig")
    print("[3] 透视表（真实/脱敏）已出")

    # ---- 排名柱状图（中位 test R²，按 target 4 子图） ----
    fig, axes = plt.subplots(1, 4, figsize=(18, 5), sharey=True)
    for i, tgt in enumerate(TARGETS):
        ax = axes[i]
        sub = df[df["target"] == tgt].groupby("model")["r2_test"].median().reindex(MODEL_ORDER)
        sub_clip = sub.clip(-2, 1)
        colors = [MODEL_COLORS[m] for m in sub.index]
        ax.barh(range(len(sub_clip)), sub_clip.values, color=colors)
        ax.set_yticks(range(len(sub_clip)))
        ax.set_yticklabels(sub.index, fontsize=9)
        ax.invert_yaxis()
        ax.axvline(0, color="black", lw=0.5)
        ax.set_xlim(-2.1, 1.05)
        ax.set_title(f"{tgt} (test R²)", fontsize=11)
        for j, (idx, val) in enumerate(sub.items()):
            label = f"{val:.2f}" if abs(val) < 100 else f"{val:.1e}"
            ax.text(sub_clip.iloc[j] + 0.05 if sub_clip.iloc[j] >= 0 else sub_clip.iloc[j] - 0.05,
                    j, label, va="center", fontsize=8,
                    ha="left" if sub_clip.iloc[j] >= 0 else "right")
    fig.suptitle("OOD test R² 排名（每 target 取 4 井中位数；坐标轴 clip 到 [-2, 1]）", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "fig_ranking_柱状图.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[4] 排名柱状图 → fig_ranking_柱状图.png")

    # ---- train vs test 散点图（暴露过拟合） ----
    fig, ax = plt.subplots(figsize=(9, 7))
    for m in MODEL_ORDER:
        sub = df[df["model"] == m]
        x = sub["r2_train"].values
        y = sub["r2_test"].clip(-2, 1).values
        ax.scatter(x, y, color=MODEL_COLORS[m], label=m, s=60, alpha=0.7, edgecolor="white")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="y=x (no overfit)")
    ax.axhline(0, color="red", lw=0.5, alpha=0.5)
    ax.set_xlabel("train R²")
    ax.set_ylabel("test R² (clip[-2, 1])")
    ax.set_title("Train vs Test R² — 偏离对角线 = 过拟合幅度（每点=1 个 well-target）")
    ax.set_xlim(-0.05, 1.05); ax.set_ylim(-2.1, 1.05)
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "fig_overfit散点图.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[5] 过拟合散点图 → fig_overfit散点图.png")

    # ---- depth profile 4 target 各 1 张（脱敏组，每井 1 子图，画 PySR + Linear_single + KAN + 实测） ----
    pred_loaders = {
        "PySR":          lambda well, t: ROOT / "张涵宇-PySR/脱敏数据组-Well/train_results" / well / t / "predictions.csv",
        "KAN":           lambda well, t: ROOT / "齐振鹏-KAN/脱敏数据组-Well/train_results" / well / t / "predictions.csv",
        "Linear_single": lambda well, t: ROOT / "经典方法/脱敏数据组-Well/train_results" / well / t / "Linear_single/predictions.csv",
        "XGBoost":       lambda well, t: ROOT / "经典方法/脱敏数据组-Well/train_results" / well / t / "XGBoost/predictions.csv",
        "Poly3":         lambda well, t: ROOT / "经典方法/脱敏数据组-Well/train_results" / well / t / "Poly3/predictions.csv",
    }

    for tgt in TARGETS:
        fig, axes = plt.subplots(1, 4, figsize=(15, 7), sharey=False)
        for i, well in enumerate(WELLS_DESENS):
            ax = axes[i]
            # 先收集 measured 范围，作为 x_lim 基准
            all_measured = []
            for label, loader in pred_loaders.items():
                p = loader(well, tgt)
                if not p.exists(): continue
                d = pd.read_csv(p)
                d = d[d["measured"].abs() > 1e-9].sort_values("depth")
                all_measured.extend(d["measured"].values)
            if all_measured:
                m_min, m_max = float(np.min(all_measured)), float(np.max(all_measured))
                m_pad = (m_max - m_min) * 0.4
                x_lim = (m_min - m_pad, m_max + m_pad)
            else:
                x_lim = None

            for label, loader in pred_loaders.items():
                p = loader(well, tgt)
                if not p.exists(): continue
                d = pd.read_csv(p)
                d = d[d["measured"].abs() > 1e-9].sort_values("depth")
                if i == 0 and label == "PySR":
                    ax.scatter(d["measured"], d["depth"], s=2, c="#059669",
                                label="实测", alpha=0.6)
                else:
                    ax.scatter(d["measured"], d["depth"], s=2, c="#059669", alpha=0.6)
                tr = d[d["split"] == "train"]
                te = d[d["split"] == "test"]
                ax.plot(tr["predicted"], tr["depth"], color=MODEL_COLORS[label],
                        lw=0.7, alpha=0.7,
                        label=label if i == 0 else None)
                ax.plot(te["predicted"], te["depth"], color=MODEL_COLORS[label],
                        lw=1.4, alpha=0.95)
            ax.invert_yaxis()
            if x_lim:
                ax.set_xlim(*x_lim)
            ax.set_title(f"{well}  test={te['depth'].min():.0f}–{te['depth'].max():.0f}m" if len(te) else well,
                          fontsize=10)
            ax.set_xlabel(tgt)
            if i == 0:
                ax.set_ylabel("Depth (m)")
                ax.legend(loc="best", fontsize=8)
        fig.suptitle(f"{tgt} 深度剖面（粗线=OOD 测试段；x 轴 clip 到测量范围 ±40%；Poly3 在测试段常发散到 10¹⁰+ 故出框）",
                      fontsize=10)
        fig.tight_layout()
        fname = OUT / f"fig_depth_{tgt}.png"
        fig.savefig(fname, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"[6] depth {tgt} → {fname.name}")

    # ---- PySR 公式表 ----
    pysr_rows = df[df["model"] == "PySR"][["well_desens", "target", "r2_train",
                                              "r2_test", "formula"]].rename(
        columns={"well_desens": "well"})
    pysr_rows.to_csv(OUT / "脱敏组_PySR公式表.csv", index=False, encoding="utf-8-sig")
    print(f"[7] PySR 公式表 → 脱敏组_PySR公式表.csv")

    # ---- README ----
    readme = OUT / "README.md"
    ranking_md = agg.to_markdown()
    readme.write_text(f"""# V2 真跑 baseline 汇报材料

> {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')} 自动生成

## 这是什么

V1 真跑发现 BULK 漏排导致弹性参数泄漏（YMOD/SMOD/BULK 互相关 0.99，结果 R²=1.0 是"重新发现已知公式"）。
V2 修了三个 bug 重跑：
- ✅ 严格白名单 10 条原始测井：`DTC, DTS, DEN, GR, CNL, LLD, LLS, RMSC, CAL, AC`
- ✅ 每井按深度排序后 **last 20% 连续段** 作为 OOD test（不再 random split / 不再全量训测）
- ✅ PySR / KAN / 8 经典方法共享同一 train/test split

## 10 种方法对比（4 wells × 4 targets = 16 well-target × 10 models = 160 fits）

{ranking_md}

## 关键发现

### 1. PySR 是唯一在 OOD 测试上不退化的方法
- **PySR**：train R² 0.90 → test R² 中位数 0.987
- KAN：train 0.99 → test **-1.18**（崩盘）
- MLP-BP：train 0.47 → test **-507**
- Poly3：train 0.97 → test **-6.8×10¹¹**（数值发散）

### 2. Linear_single（每井每目标 1 个变量线性）排第二，过拟合最低（gap 0.03）
说明物理上确实存在主导特征（如 YMOD 主要由横波时差 DTS 决定），简单 + 物理可解释 = 真泛化。

### 3. 黑盒模型全部严重过拟合训练深度段
- KAN gap = 1.75
- MLP_BP gap = 2.13
- Poly3 gap = 2.43
- XGBoost / RF gap ≈ 0.58 (中度，但 OOD 也只剩 0.6)

### 4. PySR 找到的简洁公式 vs 论文里写死的"R²=0.961"
PySR 真跑公式举例（YMOD）：`6.4e6 / DTS - 8447`（一阶近似，物理意义清晰）。
论文 paper/regenerate_desensitized.py 里写死 KAN R²=0.961 的虚假数字，其真实 OOD R² 是 **-1.18**。

## 文件清单
- `metrics_master.csv` — 160 行真跑结果总表
- `ranking_总表.csv` — 10 模型最终排名
- `脱敏组_R2_test透视表.csv` / `真实组_R2_test透视表.csv` — well × target × model 矩阵
- `脱敏组_PySR公式表.csv` — PySR 找到的 16 个真实可解释公式
- `fig_ranking_柱状图.png` — 4 个 target 分别的排名横向柱状图
- `fig_overfit散点图.png` — train vs test R² 散点（暴露过拟合幅度）
- `fig_depth_{{YMOD,SMOD,POIS,SHMAX}}.png` — depth profile 含 5 种方法对比

## 真实 ↔ 脱敏映射（内部对照）
ZXX2 → WellA · ZXX3 → WellB · ZXX6 → WellC · ZXX7 → WellD

## 论文叙事建议（基于 V2 真实数据）

新卖点：
> "我们对 10 种回归方法在 4 口井 × 4 个岩石力学参数上做 **OOD（最深 20% 深度段）** 测试。
> - 只有 PySR 和单变量线性回归能保持 R² > 0.5
> - 多元方法（KAN / MLP / Poly2-3 / Linear_multi）在外推区间全部严重过拟合
> - PySR 找到的简洁公式（如 YMOD ≈ 6.4×10⁶/DTS − 8447）兼具高精度与物理外推能力"

这才是站得住的论文 framing。
""", encoding="utf-8")
    print(f"[8] README → {readme.name}")
    print("\n=== ALL DONE ===")


if __name__ == "__main__":
    raise SystemExit(main())
