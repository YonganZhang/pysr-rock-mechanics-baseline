"""
物理范围硬过滤 + IQR 软过滤 — V2.4 双层防护

物理范围基于测井解释行业经验（Schlumberger Log Interpretation Charts）：
- 弹性模量 / 地应力：MPa 量级
- 测井曲线：仪器测量物理范围

IQR 软过滤防御未知哨兵（如井 2 的 -25400、井 4 的 -66652 等）
"""
from __future__ import annotations
import numpy as np
import pandas as pd

# ─────── 物理范围硬约束（任何超出此范围的值视为哨兵/错误）────────
# 注意：SLB 测井解释中目标变量单位不统一（YMOD=MPa, SMOD=GPa, BULK=GPa）
PHYS_RANGE = {
    # 目标变量 (rock mechanical parameters)
    "YMOD":  (1_000,    200_000),    # 杨氏模量 1000-200000 MPa（=1-200 GPa）
    "SMOD":  (0.5,      100),        # 剪切模量 0.5-100 GPa（实测数据 7-30 GPa）
    "BMOD":  (1,        200),        # 体积模量 1-200 GPa
    "BULK":  (1,        200),        # 体积模量另一记法
    "POIS":  (0.01,     0.50),       # 泊松比物理上 0~0.5
    "SHMAX": (10,       300),        # 最大水平主应力 10-300 MPa
    "SHMIN": (5,        250),
    # 原始测井曲线
    "DTC":   (30,       250),        # 纵波时差 30-250 μs/ft
    "DTS":   (50,       400),        # 横波时差
    "DEN":   (1.5,      3.5),        # 密度 1.5-3.5 g/cc
    "GR":    (0,        500),        # 自然伽马 0-500 API
    "CNL":   (-15,      80),         # 中子孔隙度 -15~80%
    "LLD":   (0.01,     50_000),     # 深侧向电阻率
    "LLS":   (0.01,     50_000),
    "RMSC":  (0.01,     5_000),      # 微聚焦电阻率
    "CAL":   (4,        30),         # 井径 4-30 inch
    "PEF":   (0.5,      10),         # 光电吸收
    "AC":    (30,       300),        # 声波 (旧记法，等同 DTC)
    "DEPTH": (0,        20_000),     # 深度 0-20km
}


def physical_range_mask(df: pd.DataFrame, columns: list) -> pd.Series:
    """对给定列做物理范围过滤，返回有效行 mask（True=保留）"""
    mask = pd.Series(True, index=df.index)
    for c in columns:
        if c not in df.columns: continue
        if c not in PHYS_RANGE: continue
        v = pd.to_numeric(df[c], errors="coerce")
        lo, hi = PHYS_RANGE[c]
        col_mask = (v >= lo) & (v <= hi)
        mask &= col_mask
    return mask


def iqr_outlier_mask(df: pd.DataFrame, columns: list, k: float = 5.0) -> pd.Series:
    """IQR 软过滤：超出 [q01 - k*IQR, q99 + k*IQR] 视为异常 (k=5 较宽松防误杀)"""
    mask = pd.Series(True, index=df.index)
    for c in columns:
        if c not in df.columns: continue
        v = pd.to_numeric(df[c], errors="coerce").dropna()
        if len(v) < 50: continue
        q01, q99 = v.quantile(0.01), v.quantile(0.99)
        iqr = q99 - q01
        if iqr <= 0: continue
        lo, hi = q01 - k * iqr, q99 + k * iqr
        col_v = pd.to_numeric(df[c], errors="coerce")
        col_mask = ((col_v >= lo) & (col_v <= hi)) | col_v.isna()
        mask &= col_mask
    return mask


def clean_well_data(df_well: pd.DataFrame, target: str, features: list,
                     verbose: bool = False) -> pd.DataFrame:
    """完整清洗：dropna + 物理范围 + IQR + |target|>1e-9"""
    cols = ["DEPTH", target] + features
    used = df_well[cols].copy()
    for c in cols:
        used[c] = pd.to_numeric(used[c], errors="coerce")
    used = used.replace([np.inf, -np.inf], np.nan).dropna()
    n0 = len(used)
    # 1. 物理范围硬过滤
    used = used[physical_range_mask(used, cols)]
    n1 = len(used)
    # 2. IQR 软过滤
    used = used[iqr_outlier_mask(used, cols, k=5.0)]
    n2 = len(used)
    # 3. target |x| > 1e-9
    used = used[used[target].abs() > 1e-9].reset_index(drop=True)
    n3 = len(used)
    if verbose:
        print(f"  cleaning: {n0} → {n1} (phys) → {n2} (iqr) → {n3} (target>0)")
    return used


if __name__ == "__main__":
    # 测试：dump 每井每目标清洗前后 SHMAX 范围
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] /
                          "学生-张涵宇-PYSR改进/.trae/skills/pysr_equation_fit"))
    from a import build_dataset_from_dir
    from pathlib import Path

    df = build_dataset_from_dir(Path("学生-张涵宇-PYSR改进/data"), keep_common_columns_only=False)
    df.loc[df["WELLNAME"]=="ZXX3井","WELLNAME"]="ZXX3"
    df.loc[df["WELLNAME"]=="ZXX7井","WELLNAME"]="ZXX7"
    df.loc[df["WELLNAME"]=="足212（导眼井）","WELLNAME"]="ZXX2"

    WHITE = ["DTC","DTS","DEN","GR","CNL","LLD","LLS","RMSC","CAL"]
    for w in ["ZXX2","ZXX3","ZXX6","ZXX7"]:
        for t in ["YMOD","SMOD","POIS","SHMAX"]:
            sub = df[df["WELLNAME"]==w]
            if t not in sub.columns: continue
            print(f"\n{w}/{t}:")
            cleaned = clean_well_data(sub, t, WHITE, verbose=True)
            if len(cleaned) > 0:
                print(f"  → {t} range: [{cleaned[t].min():.2f}, {cleaned[t].max():.2f}]"
                      f"  mean={cleaned[t].mean():.2f}  std={cleaned[t].std():.2f}")
