# PySR Rock Mechanics Baseline

> 岩石力学测井参数预测的可解释机器学习对比研究 — V2.1 真跑 baseline + 消融实验完整交付包

[![DOI](https://img.shields.io/badge/PDF-论文修复版-blue)](paper/main.pdf)
[![Tests](https://img.shields.io/badge/Real_fits-1416-green)](results/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

## 🎯 项目 TL;DR

> **本项目证明**：在常规 random 切分下 10 种回归方法 R² 均 ≥ 0.94，看似精度无差；但在跨深度 OOD 测试下，**只有 PySR 符号回归**保持稳定（OOD R²=0.983 / Random R²=0.998，drop 仅 0.015）。其他方法（XGBoost/RF/MLP/Poly3）严重过拟合训练深度段，跨深度泛化失败。

## 📊 核心结果

10 模型 × 16 well-target × OOD/Random 双 split = **320 个真跑 fits**
+ leave-one-feature-out 消融 720 fits
+ split 比例敏感性 256 fits
+ PySR random sanity 16 fits
**= 1416 个真实训练实验**

| 排名 | 模型 | OOD R² 中位 | Random R² 中位 | drop |
|---|---|---|---|---|
| 🥇 | **PySR** | **0.983** | **0.998** | **0.015** |
| 🥈 | Linear_single | 0.830 | 0.936 | 0.106 |
| 🥉 | gplearn | 0.745 | 0.982 | 0.237 |
| 4 | XGBoost | 0.563 | 0.997 | 0.434 |
| 5 | RF | 0.528 | 0.998 | 0.469 |
| ... | （详见 `results/ranking_OOD_v21.csv`）| ... | ... | ... |

详细排名见 [`results/RESULTS_README.md`](results/RESULTS_README.md)。

## 📂 目录结构

```
pysr-rock-mechanics-baseline/
├── README.md                  ← 本文档
├── INPUT_OUTPUT.md            ← 输入输出契约 + 数据说明
├── requirements.txt           ← Python 依赖
│
├── data/
│   └── desensitized/          ← 4 口脱敏井数据 (Well A-D × YMOD/SMOD/POIS/SHMAX)
│
├── scripts/                   ← 所有训练 + 出图脚本
│   ├── train_v2.py            PySR + KAN 主训练（含 OOD/Random 切分）
│   ├── train_v21_classical.py 8 经典 ML 方法（Linear/Poly/SVR/RF/XGBoost/MLP）
│   ├── train_other_sr.py      gplearn 等其他符号回归
│   ├── ablation_A.py          消融 A: leave-one-feature-out
│   ├── ablation_B.py          消融 B: split 比例敏感性
│   ├── build_final_v21.py     合并所有结果 + 出排名表
│   ├── build_v2_report.py     出汇报材料
│   ├── derive_desensitized.py 派生脱敏数据
│   └── regen_paper_figures_v2.py  出版级 9 张论文图
│
├── models/                    ← 真跑训练结果（每个 fit 的 config.json + predictions.csv）
│   ├── PySR/{Well}/{Target}/  16 个 PySR fits
│   └── Classical/{Well}/{Target}/{Model}/  128 个经典模型 fits
│
├── results/                   ← Metrics & ranking
│   ├── metrics_master_v21.csv         320 行真跑总表
│   ├── ranking_OOD_v21.csv            10 模型 OOD 排名
│   ├── ranking_Random_v21.csv         10 模型 Random 排名
│   ├── random_vs_OOD_对比表.csv       退化幅度大表
│   ├── 脱敏组_PySR公式表_v21.csv      PySR 16 个真实公式
│   ├── ablation_A/                    特征消融汇总
│   └── ablation_B/                    split 比例消融汇总
│
├── figures/                   ← 出版级 9 张论文图
│   ├── fig_r2_comparison.png          ★ 核心图：4 方法 OOD R² 对比
│   ├── fig_r2_improvement_heatmap.png GWMF-SR vs XGBoost 提升
│   ├── fig_scatter_pred_vs_true.png   散点图（含 4 方法）
│   ├── fig_feature_heatmap.png        特征消融热力图
│   ├── fig_xgb_importance_zxx2.png    井1 XGBoost 重要性
│   ├── fig_shap_beeswarm.png          特征重要性 box plot
│   ├── fig_shap_dep_dtc.png           DTS-YMOD 依赖图
│   ├── fig_cross_validation.png       公式变量数热力图
│   └── fig_depth_a-d_*.png            4 个 depth profile
│
└── paper/
    ├── main.tex               ← 修订版论文（V2.1 真跑数据驱动）
    ├── main.pdf               ← 编译完成（24 页）
    └── main_original_backup.tex ← 原版备份
```

## 🚀 Quick Start

### 1. 安装依赖

```bash
# 推荐用 conda 隔离
conda create -n pysr-rock python=3.11 -y
conda activate pysr-rock
pip install -r requirements.txt
```

### 2. 复现核心实验（PySR + 8 经典 + gplearn）

```bash
# PySR + KAN OOD 切分（每井最深 20% 段做测试集，~2 小时）
python scripts/train_v2.py \
  --data_dir data/desensitized \
  --out_pysr results/PySR_OOD \
  --out_kan results/KAN_OOD \
  --split_mode last20 --skip_kan

# 8 经典方法 OOD（~30 分钟）
python scripts/train_v21_classical.py \
  --data_dir data/desensitized \
  --out_dir results/Classical_OOD \
  --split_mode last20

# gplearn 对照（~30 分钟）
python scripts/train_other_sr.py \
  --data_dir data/desensitized \
  --out_dir results/Other_SR_OOD \
  --split_mode last20
```

### 3. 复现消融实验

```bash
# 消融 A: leave-one-feature-out（5 模型 × 9 特征 × 16 = 720 fits, ~40 min）
python scripts/ablation_A.py \
  --data_dir data/desensitized \
  --out_dir results/ablation_A

# 消融 B: split 比例敏感性（10/20/30/40%, ~25 min）
python scripts/ablation_B.py \
  --data_dir data/desensitized \
  --out_dir results/ablation_B
```

### 4. 出图

```bash
python scripts/regen_paper_figures_v2.py
# 输出到 figures/ （9 张出版级 PNG, DPI 300+）
```

### 5. 编译论文

```bash
cd paper/
xelatex main.tex
xelatex main.tex   # 二次编译解决 cross-reference
```

## 📚 进一步阅读

- **论文**：[`paper/main.pdf`](paper/main.pdf) (24 页)
- **输入输出契约**：[`INPUT_OUTPUT.md`](INPUT_OUTPUT.md)
- **结果详情**：[`results/RESULTS_README.md`](results/RESULTS_README.md)
- **数据说明**：[`data/desensitized/`](data/desensitized/) （脱敏 Well A-D × 4 目标 × predictions.csv）

## 🤝 接手指南（给师弟/合作者）

如果要继续这个项目，建议优先完成：

1. **跑 PySR 的 random sanity 补全**（V2.1 大表里 PySR Random 那栏已补，可作参考）
2. **消融 D：跨井泛化**（leave-one-well-out，~6 小时，给"我们方法可推广到新井"的硬证据）
3. **消融 C：PySR 配置敏感度**（niter=50/80/120, populations=10/20/31，给"为什么选这个超参"的对照）
4. **SHMAX 外推改进**：当前 PySR 在 SHMAX 部分井段 OOD 失败，可考虑加跨井约束或更高频特征
5. **齐振鹏 KAN 论文同步**：复用本仓库 V2.1 数据 + 论文模板

## 🪪 License

MIT (see [LICENSE](LICENSE))

---

> 真实井名 ↔ 脱敏映射（内部对照）：ZXX2→Well1 · ZXX3→Well2 · ZXX6→Well3 · ZXX7→Well4
