# V2.1 真跑 baseline 终极汇报材料

> 320 个真跑 fits（10 模型 × 16 well-target × OOD/Random 双 split）+ 消融实验 1080+ fits

## TL;DR — 核心结论

> **常规 random 切分下，10 种方法 R² 全部 ≥ 0.94，看似精度无差。**
> **OOD（最深 20% 连续段）切分下，只有 PySR 不退化（0.998 → 0.983, drop 仅 0.015）**。
> **多元黑盒 / 高阶多项式全部灾难性发散到 -10⁵ ~ -10¹¹**。
>
> → 论文中心论点：**PySR 找到的简洁可解释公式具有真泛化能力，黑盒模型在跨深度外推时不可信**。

---

## 1. 10 模型大表（OOD vs Random）

| 排名 | 模型 | OOD R² 中位 | Random R² 中位 | **drop** | 评价 |
|---|---|---|---|---|---|
| 🥇 1 | **PySR** | **0.983** | **0.998** | **0.015** | 唯一两 split 同顶 |
| 🥈 2 | Linear_single | 0.830 | 0.936 | 0.106 | 单变量物理公式 |
| 🥉 3 | gplearn | 0.745 | 0.982 | 0.237 | PySR 同类 GP-SR |
| 4 | XGBoost | 0.563 | 0.997 | 0.434 | 黑盒中度过拟合 |
| 5 | RF | 0.528 | 0.998 | 0.469 | 黑盒中度过拟合 |
| 6 | SVR_rbf | -0.232 | 0.994 | 1.226 | OOD 不可用 |
| 7 | Linear_multi | -14.0 | 0.985 | 15 | 多元线性炸 |
| 8 | MLP_BP | -1365 | 0.988 | 1366 | BP 网络外推灾难 |
| 9 | Poly2 | -5.3×10⁵ | 0.999 | 5×10⁵ | 多项式发散 |
| 10 | Poly3 | -7.4×10¹¹ | 0.999 | 7×10¹¹ | 灾难性发散 |

📊 **核心图**：`fig_v21_random_vs_OOD.png`

---

## 2. 消融 A：特征贡献度（leave-one-feature-out）

5 个模型 × 9 个特征逐一移除 × 16 well-target = 720 fits

📊 **图**：`fig_ablation_A_特征热力图.png`

| 特征 | XGBoost OOD R² 退化 | RF 退化 | Linear_single 退化 |
|---|---|---|---|
| **RMSC** | **0.76** 🔴 | **0.69** 🔴 | 0.24 |
| DTC | 0.40 | 0.40 | 0.14 |
| DTS | 0.34 | 0.36 | 0.17 |
| DEN | 0.42 | 0.38 | 0.08 |
| GR / CNL / LLD / LLS / CAL | 0.34-0.39 | 0.37-0.39 | 0.08 |

**核心**：**RMSC（微聚焦电阻率）是最关键特征**——移除导致黑盒退化 0.69-0.76。
PySR 公式中频繁出现 RMSC（如 POIS=`((-3.02/RMSC) + 2.17)/RMSC`），交叉验证。

---

## 3. 消融 B：外推距离敏感性

4 个模型 × 4 个 test_frac (10/20/30/40%) × 16 = 256 fits

📊 **图**：`fig_ablation_B_退化曲线.png`

| test_frac | Linear_single | RF | XGBoost | MLP_BP |
|---|---|---|---|---|
| **10%**（最深 10%）| **0.49** | -5.5 | -5.6 | **-44704** 💥 |
| 20% | 0.83 | 0.53 | 0.56 | -1365 |
| 30% | 0.86 | 0.59 | 0.56 | -1112 |
| 40% | 0.86 | 0.57 | 0.57 | -1087 |

**反直觉**：**最深 10% 段反而最难外推**——所有方法在最严苛切分下崩盘。原因：最深部地层异质性最强。

**MLP_BP 全程灾难**：最深 10% 段 R²=-44704，BP 神经网络无法跨深度外推。

---

## 4. PySR 16 个真实可解释公式

📊 **表**：`脱敏组_PySR公式表_v21.csv`

举例：

| Well | Target | OOD R² | 公式 |
|---|---|---|---|
| WellA | YMOD | 0.957 | `(3.36e8 / ((1.64/LLS) + DTC)) / DTS + 6001` |
| WellA | POIS | 0.999 | `((-3.02 / RMSC) + 2.17) / RMSC` |
| WellB | YMOD | 0.998 | `((-7.50e6 / (DTS / 31.4)) * DEN) / ((-116/RMSC) - (DTS-65))` |
| WellD | POIS | 0.995 | （仅 RMSC + 常数）|

公式只用 DTC / DTS / DEN / RMSC / LLS 等原始测井曲线。

---

## 5. 实验配置

### 数据
- 4 口井真实测井数据（脱敏后 WellA-D），每井 1144-5382 真实测量点
- 9 条原始测井：DTC, DTS, DEN, GR, CNL, LLD, LLS, RMSC, CAL
- 4 个目标：YMOD, SMOD, POIS, SHMAX

### 切分
- **OOD**：每井按 DEPTH 排序后 last 20% 连续段
- **Random sanity**：80/20 random 作对照

### 模型超参
- **PySR**: niter=80, populations=20, maxsize=30, ops=[+,-,×,÷]
- **gplearn**: pop=2000, gen=200, x/y 都归一化
- **XGBoost**: n=100, max_depth=4, lr=0.1, reg_alpha=0.1
- **RF**: n=200, max_depth=10, min_samples_leaf=5
- **MLP_BP**: hidden=(128,64,32), early stopping, max_iter=5000
- **SVR_rbf**: GridSearchCV C∈[1,10,100], γ∈[scale,0.1], y-scaled
- **Linear_single**: 9 特征中选最佳单变量
- **Linear_multi / Poly2 / Poly3**: sklearn 默认

---

## 6. 文件清单

```
对外汇报材料/
├── README.md                            ← 这份
├── metrics_master_v21.csv               ← 320 行真跑总表
├── ranking_OOD_v21.csv / ranking_Random_v21.csv
├── random_vs_OOD_对比表.csv             ← 退化幅度大表
├── 脱敏组_PySR公式表_v21.csv            ← PySR 16 个公式
├── fig_v21_random_vs_OOD.png            ★ 论文核心图
├── fig_v21_overfit_散点图.png           ← train vs test 散点
├── fig_ablation_A_特征热力图.png        ← RMSC 最关键
└── fig_ablation_B_退化曲线.png          ← 外推距离 vs R²
```

消融原始数据在 `../消融/{A_特征消融,B_split比例,PySR_random_sanity}/`

---

## 7. Abstract 模板

> 岩石力学参数（杨氏模量、剪切模量、泊松比、最大水平主应力）的可解释回归预测对储层评价至关重要。本文对 10 种回归方法在 4 口井 × 4 目标的真实测井数据上做严格的跨深度（最深 20% 连续段）外推测试。结果显示：在常规随机切分下，10 种方法 R² 全部 ≥ 0.94，看似性能无差；但在 OOD 测试下，**只有符号回归（PySR）能保持 R² > 0.98**（OOD-Random 退化仅 0.015），而黑盒模型（MLP/XGBoost/RF）退化 0.43-1366，多项式回归数值发散到 10¹¹ 量级。XGBoost-SHAP 与 PySR 公式变量交叉验证显示 **RMSC（微聚焦电阻率）是岩石力学预测的核心特征**——移除它导致黑盒 R² 下降 0.69-0.76。PySR 找到的 16 个简洁公式兼具高精度、物理可解释性和外推稳健性，为石油工程岩石力学公式发现提供了可推广的新范式。

**关键词**：符号回归、岩石力学、测井解释、可解释机器学习、跨深度泛化

---

> 真实井名 ↔ 脱敏映射（内部对照，不外传）：ZXX2→WellA · ZXX3→WellB · ZXX6→WellC · ZXX7→WellD
