"""
其他符号回归 (SR) 方法对照 — 替代 KAN 跟 PySR 比较
- gplearn: 经典 GP-based SR (Python sklearn-style)
- pyoperon: C++ 高性能 GP-SR

跟 PySR 同 split / 同 9 特征 / 同 OOD test。
输出格式跟 train_v21_classical.py 一致。
"""
from __future__ import annotations
import argparse, json, sys, time, warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
import train_v2 as tv2
import train_v21_classical as tv21

from gplearn.genetic import SymbolicRegressor as GPLearnSR
from sklearn.linear_model import Lasso, LassoCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures


def _r2(y, yp):
    ss_res = float(np.sum((y - yp) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def fit_gplearn(d):
    """gplearn: 增大种群 + generations 200，y/X 都归一化"""
    x_scaler = StandardScaler().fit(d["X_train"])
    Xn_tr, Xn_te = x_scaler.transform(d["X_train"]), x_scaler.transform(d["X_test"])
    y_scaler = StandardScaler().fit(d["y_train"].reshape(-1, 1))
    yn = y_scaler.transform(d["y_train"].reshape(-1, 1)).ravel()
    m = GPLearnSR(
        population_size=2000, generations=200,
        function_set=("add", "sub", "mul", "div"),
        metric="mse", parsimony_coefficient=0.0001,
        const_range=(-3.0, 3.0),
        p_crossover=0.7, p_subtree_mutation=0.1, p_hoist_mutation=0.05,
        p_point_mutation=0.1, p_point_replace=0.05,
        max_samples=1.0, verbose=0, n_jobs=8, random_state=42,
    )
    m.fit(Xn_tr, yn)
    yp_tr_n, yp_te_n = m.predict(Xn_tr), m.predict(Xn_te)
    yp_tr = y_scaler.inverse_transform(yp_tr_n.reshape(-1, 1)).ravel()
    yp_te = y_scaler.inverse_transform(yp_te_n.reshape(-1, 1)).ravel()
    formula = str(m._program)
    for i, fn in enumerate(d["features"]):
        formula = formula.replace(f"X{i}", fn)
    return {"yp_tr": yp_tr, "yp_te": yp_te, "formula": f"(scaled) {formula[:180]}"}


def fit_lasso_poly(d):
    """LASSO + 多项式 deg=2 交互特征 = sparse interpretable formula
    本质：先生成 X, X*X, X*Y, ... 多项式特征，再 LASSO 自动选择稀疏子集
    输出公式可读：y = c1*DTC + c2*DTS² + c3*DTC*DEN + ...
    """
    pf = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
    x_scaler = StandardScaler()
    Xp_tr = pf.fit_transform(x_scaler.fit_transform(d["X_train"]))
    Xp_te = pf.transform(x_scaler.transform(d["X_test"]))
    m = LassoCV(cv=3, max_iter=10000, n_alphas=30, n_jobs=8, random_state=42)
    m.fit(Xp_tr, d["y_train"])
    yp_tr, yp_te = m.predict(Xp_tr), m.predict(Xp_te)
    # 提取非零项
    feat_names = pf.get_feature_names_out([f"x{i}" for i in range(d["X_train"].shape[1])])
    nz_idx = np.where(np.abs(m.coef_) > 1e-8)[0]
    terms = []
    for i in nz_idx:
        term = feat_names[i]
        for j, fn in enumerate(d["features"]):
            term = term.replace(f"x{j}", fn)
        terms.append(f"{m.coef_[i]:+.3g}*{term}")
    formula = " ".join(terms[:8]) + (f" ... ({len(nz_idx)} nz)" if len(nz_idx)>8 else "") + f" {m.intercept_:+.3g}"
    return {"yp_tr": yp_tr, "yp_te": yp_te, "formula": formula[:250]}


def fit_lasso_poly3(d):
    """同上但 degree=3（更复杂的可解释公式）"""
    pf = PolynomialFeatures(degree=3, include_bias=False, interaction_only=False)
    x_scaler = StandardScaler()
    Xp_tr = pf.fit_transform(x_scaler.fit_transform(d["X_train"]))
    Xp_te = pf.transform(x_scaler.transform(d["X_test"]))
    m = LassoCV(cv=3, max_iter=10000, n_alphas=30, n_jobs=8, random_state=42)
    m.fit(Xp_tr, d["y_train"])
    yp_tr, yp_te = m.predict(Xp_tr), m.predict(Xp_te)
    nz = int(np.sum(np.abs(m.coef_) > 1e-8))
    return {"yp_tr": yp_tr, "yp_te": yp_te, "formula": f"LassoPoly3 (sparse, n_nz={nz}/{len(m.coef_)})"}


def fit_operon(d):
    """pyoperon: float32（pyoperon FitLeastSquares 要求 float32 一致）"""
    Xt = np.ascontiguousarray(d["X_train"], dtype=np.float32)
    yt = np.ascontiguousarray(d["y_train"], dtype=np.float32)
    Xv = np.ascontiguousarray(d["X_test"], dtype=np.float32)
    m = OperonSR(
        allowed_symbols="add,sub,mul,div,constant,variable",
        population_size=1000,
        generations=200,
        max_length=30,
        max_depth=10,
        n_threads=8,
        random_state=42,
    )
    m.fit(Xt, yt)
    yp_tr = m.predict(Xt).astype(np.float64)
    yp_te = m.predict(Xv).astype(np.float64)
    try:
        formula = m.get_model_string(m.model_, precision=3)
        for i, fn in enumerate(d["features"]):
            formula = formula.replace(f"X{i}", fn)
    except Exception:
        formula = "Operon_GP"
    return {"yp_tr": yp_tr, "yp_te": yp_te, "formula": formula[:200]}


MODELS = {"gplearn": fit_gplearn}


def run_one(d, well, target, out_dir):
    rows = []
    for name, fn in MODELS.items():
        sub_dir = out_dir / tv2.safe_dir_name(well) / tv2.safe_dir_name(target) / name
        if (sub_dir / "config.json").exists():
            print(f"  {name:10s} SKIP (已存在)")
            continue
        sub_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        try:
            r = fn(d)
            elapsed = time.time() - t0
            r2_tr = _r2(d["y_train"], r["yp_tr"])
            r2_te = _r2(d["y_test"], r["yp_te"])
            mae_te = float(np.mean(np.abs(d["y_test"] - r["yp_te"])))
            pred_df = pd.DataFrame({
                "depth": np.concatenate([d["depth_train"], d["depth_test"]]),
                "measured": np.concatenate([d["y_train"], d["y_test"]]),
                "predicted": np.concatenate([r["yp_tr"], r["yp_te"]]),
                "split": ["train"] * d["n_train"] + ["test"] * d["n_test"],
            }).sort_values("depth")
            pred_df.to_csv(sub_dir / "predictions.csv", index=False, encoding="utf-8-sig")
            cfg = {"well": well, "target": target, "model": name,
                   "split_mode": d["split_mode"],
                   "n_train": d["n_train"], "n_test": d["n_test"],
                   "features": d["features"], "n_features": len(d["features"]),
                   "elapsed_sec": round(elapsed, 2),
                   "r2_train": round(float(r2_tr), 4),
                   "r2_test": round(float(r2_te), 4),
                   "mae_test": round(float(mae_te), 4),
                   "formula_or_arch": r.get("formula", name)}
            (sub_dir / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2),
                                                  encoding="utf-8")
            print(f"  {name:10s} train={r2_tr:.4f} test={r2_te:.4f} ({elapsed:.0f}s)")
            sys.stdout.flush()
            rows.append({"well": well, "target": target, "model": name,
                         "r2_train": round(float(r2_tr), 4),
                         "r2_test": round(float(r2_te), 4),
                         "elapsed_sec": round(elapsed, 2),
                         "formula": r.get("formula", name)[:150]})
        except Exception as e:
            print(f"  {name:10s} ERROR: {e}")
            sys.stdout.flush()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, type=Path)
    ap.add_argument("--out_dir", required=True, type=Path)
    ap.add_argument("--split_mode", default="last20", choices=["last20", "random"])
    ap.add_argument("--targets", default=",".join(tv2.TARGETS))
    args = ap.parse_args()
    targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Other SR (gplearn + pyoperon) split={args.split_mode} ===")
    print(f"Models: {list(MODELS.keys())}")
    sys.stdout.flush()
    df = tv2.load_real_data(args.data_dir)
    wells = sorted({str(x) for x in df["WELLNAME"].dropna().unique() if str(x).strip()})
    rows = []; t0 = time.time()
    for well in wells:
        df_well = df[df["WELLNAME"] == well].copy()
        for target in targets:
            d = tv21.prepare_v21(df_well, target, split_mode=args.split_mode)
            if d is None: continue
            print(f"\n{well}/{target}: train={d['n_train']} test={d['n_test']}")
            sys.stdout.flush()
            rows.extend(run_one(d, well, target, args.out_dir))
    pd.DataFrame(rows).to_csv(args.out_dir / "metrics_summary.csv",
                               index=False, encoding="utf-8-sig")
    print(f"\n=== {time.time()-t0:.0f}s 总，summary 保存 ===")


if __name__ == "__main__":
    raise SystemExit(main())
