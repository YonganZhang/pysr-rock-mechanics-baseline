# Input / Output 契约说明

> 本文档详细解释：模型的输入是什么、输出是什么、数据格式约定、训练/测试切分方式。

## 1. 输入特征（9 条原始测井曲线）

模型严格只用 **仪器直接测量的原始测井曲线**，不使用任何衍生量（如弹性参数、脆性指数、矿物含量等）。

| 特征 | 全名 | 物理意义 | 单位 |
|---|---|---|---|
| `DTC` | Compressional sonic | 纵波时差 | μs/ft |
| `DTS` | Shear sonic | 横波时差 | μs/ft |
| `DEN` | Bulk density | 体积密度 | g/cm³ |
| `GR` | Natural gamma ray | 自然伽马 | API |
| `CNL` | Compensated neutron | 中子孔隙度 | v/v |
| `LLD` | Deep laterolog | 深侧向电阻率 | Ω·m |
| `LLS` | Shallow laterolog | 浅侧向电阻率 | Ω·m |
| `RMSC` | Micro-spherically focused resistivity | 微聚焦电阻率 | Ω·m |
| `CAL` | Caliper | 井径 | inch |

**为什么严格白名单**：原始测井数据中可能包含已计算的弹性参数（YMOD/SMOD/BULK 等），如果作为输入会导致目标变量泄漏（数据 leakage），R² 虚高至接近 1.0 但毫无知识发现意义。本项目通过 `train_v2.py` 中 `WHITELIST` 常量强制只用上述 9 条原始曲线。

## 2. 输出目标（4 个岩石力学参数）

| 目标 | 全名 | 物理意义 | 单位 |
|---|---|---|---|
| `YMOD` | Young's modulus | 杨氏模量（刚度）| MPa |
| `SMOD` | Shear modulus | 剪切模量（抗剪）| MPa |
| `POIS` | Poisson's ratio | 泊松比（应变比）| (无量纲) |
| `SHMAX` | Maximum horizontal stress | 最大水平主应力 | MPa |

## 3. 数据切分（OOD 严格评估）

```
对每口井独立处理：
  1. 按 DEPTH 升序排列
  2. 取最深 20% 连续段作为 OOD test set
  3. 其余 80%（浅至中部）作为 train set
```

**为什么是 last 20% by depth 而不是 random 80/20**：
- 测井数据沿深度有强自相关性（相邻样点高度相似）
- Random 切分会让训练集和测试集深度交错，相当于"在测试集附近见过相似样本"，无法评估真实外推能力
- Last 20% by depth 保证测试段对应的地层深度在训练阶段从未出现，是严苛的 distribution shift 测试

## 4. 数据预处理流程

```python
# train_v2.py prepare_well_target():
1. df.dropna()                          # 去 NaN 行
2. df[target.abs() > 1e-9]              # 去填充零值（浅部假数据）
3. sort by DEPTH                        # 按深度排序
4. last 20% → test, first 80% → train   # OOD 切分
```

**KAN 训练时**（已不是主对比，仅作 V2 历史参考）：
- X/y 用训练集均值/方差归一化（test 用 train 统计量）

**PySR 训练时**：
- 不归一化（PySR 直接处理原始量纲，输出公式保留物理单位）

**经典方法**（XGBoost/RF/MLP/SVR）：
- 在 sklearn pipeline 内嵌 StandardScaler

## 5. 模型超参数

### PySR (GWMF-SR 包装)

```python
PySRRegressor(
    niterations=80,
    populations=20,
    maxsize=30,
    binary_operators=["+", "-", "*", "/"],
    unary_operators=[],         # 严格只用 4 个二元算子
    random_state=42,
    deterministic=False,
)
```

### XGBoost
```python
XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1,
             reg_alpha=0.1, reg_lambda=1.0, n_jobs=8, random_state=42)
```

### Random Forest
```python
RandomForestRegressor(n_estimators=200, max_depth=10,
                      min_samples_leaf=5, n_jobs=8, random_state=42)
```

### MLP_BP
```python
MLPRegressor(hidden_layer_sizes=(128, 64, 32),
             activation="relu", solver="adam",
             max_iter=5000, random_state=42,
             learning_rate="adaptive",
             early_stopping=True, alpha=1e-4)
```

### SVR (RBF kernel)
```python
GridSearchCV(SVR(kernel="rbf"),
             param_grid={"C": [1, 10, 100], "gamma": ["scale", 0.1]},
             cv=3, n_jobs=8)
# y 也归一化（修小尺度变量崩盘）
```

### Linear / Polynomial baseline
- Linear_single：每井每目标遍历 9 特征单独线性回归，取最佳 R²
- Linear_multi：9 特征同时做多元线性回归
- Poly2/Poly3：StandardScaler + PolynomialFeatures(degree=2/3) + Ridge(α=1.0)

## 6. 输出格式

每个 well-target × model 组合输出：

```
{out_dir}/{Well}/{Target}/{Model}/
├── config.json          # well, target, n_train, n_test, r2_train, r2_test, mae_test, formula, ...
└── predictions.csv      # depth, measured, predicted, split (train/test)
```

**config.json schema**:
```json
{
  "well": "ZXX2",
  "target": "YMOD",
  "model": "PySR",
  "split_mode": "last20",
  "n_train": 2140,
  "n_test": 535,
  "depth_train_range": [4023.0, 4236.9],
  "depth_test_range": [4237.0, 4290.4],
  "features": ["DTC", "DTS", ..., "CAL"],
  "n_features": 9,
  "r2_train": 0.84,
  "r2_test": 0.957,
  "mae_test": 1234.5,
  "formula": "(3.36e8 / ((1.64/LLS) + DTC)) / DTS + 6001",
  "elapsed_sec": 87.3
}
```

**predictions.csv schema**:
```
depth,measured,predicted,split
4023.1,42850.3,42100.5,train
4023.2,42990.8,42305.2,train
...
4237.1,55400.0,55812.3,test
...
```

## 7. 评估指标

```python
# train_v2.py
def _r2_score(y_true, y_pred):
    ss_res = sum((y_true - y_pred) ** 2)
    ss_tot = sum((y_true - y_true.mean()) ** 2)
    return 1 - ss_res / max(ss_tot, 1e-12)
```

**注意**：当 OOD 测试集的 y 分布与训练集差异极大时，R² 可能为负数（甚至发散到 -10¹¹）。这反映了模型在 OOD 区间无法外推。本项目允许负 R²，并在汇总表中用 clip [-2, 1] 做可视化（同时保留原始数值在 csv 中）。

## 8. 真跑结果可复现性

所有 random_state=42 固定，重跑应得到相同结果（除 PySR 默认 deterministic=False，但中位数 R² 应稳定）。

如需完全确定的 PySR 结果，在 `train_v2.py` 中改 `deterministic=True`。

## 9. 数据脱敏映射

为保密原因，公开版本数据用脱敏井名：

| 真实 | 脱敏 |
|---|---|
| ZXX2 | Well 1 (WellA) |
| ZXX3 | Well 2 (WellB) |
| ZXX6 | Well 3 (WellC) |
| ZXX7 | Well 4 (WellD) |

数据本身（DEPTH 范围、measured 数值）保持原样，仅井名替换。
