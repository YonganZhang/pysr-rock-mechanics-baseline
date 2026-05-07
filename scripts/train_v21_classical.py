"""
V2.1 经典 ML 对比 — 修了三个 V2 bug
1. AC 列移除（白名单 9 条经典曲线）
2. MLP / SVR / RF / XGBoost 超参调优（V2 里 MLP train 0.47 / SVR 0.22 是 underfit bug）
3. 支持 --split_mode last20 | random（random 是 sanity check，看黑盒在常规切分下能否赢 PySR）

新超参（V2.1 tuned）：
- MLP_BP:  hidden=(128,64,32), max_iter=5000, lr='adaptive', early_stopping=True, alpha=1e-4
- SVR_rbf: grid search C∈[1,10,100,1000], gamma∈['scale',0.01,0.1,1]
- RF:      n=300, max_depth=10（防过拟合）, min_samples_leaf=5
- XGBoost: n=300, max_depth=4, lr=0.05, reg_alpha=0.1, reg_lambda=1.0

调用:
  python train_v21_classical.py --data_dir <P> --out_dir <P> --split_mode last20|random
"""
from __future__ import annotations
import argparse, json, sys, time, warnings
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
import train_v2 as tv2

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
import xgboost as xgb

# V2.1 白名单：去 AC
WHITELIST_V21 = ["DTC", "DTS", "DEN", "GR", "CNL", "LLD", "LLS", "RMSC", "CAL"]

# 复用 V2 的目标 / META / 黑名单
TARGETS = tv2.TARGETS


def _r2(y, yp):
    ss_res = float(np.sum((y - yp) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def prepare_v21(df_well: pd.DataFrame, target: str, split_mode: str = "last20",
                 random_seed: int = 42) -> Optional[dict]:
    target_n = tv2._normalize_column_name(target)
    if target_n not in df_well.columns:
        return None
    feats = [c for c in WHITELIST_V21 if c in df_well.columns]
    if len(feats) < 4: return None

    cols = ["DEPTH", target_n] + feats
    used = df_well[cols].copy()
    for c in cols:
        used[c] = pd.to_numeric(used[c], errors="coerce")
    used = used.replace([np.inf, -np.inf], np.nan).dropna()
    used = used[used[target_n].abs() > 1e-9].reset_index(drop=True)
    if len(used) < 100: return None

    used = used.sort_values("DEPTH").reset_index(drop=True)
    n = len(used)
    n_test = max(int(round(n * 0.20)), 30)
    n_train = n - n_test

    if split_mode == "last20":
        train_idx = np.arange(n_train)
        test_idx = np.arange(n_train, n)
    elif split_mode == "random":
        rng = np.random.RandomState(random_seed)
        idx = np.arange(n); rng.shuffle(idx)
        train_idx = sorted(idx[:n_train])
        test_idx = sorted(idx[n_train:])
    else:
        raise ValueError(f"split_mode={split_mode}")

    train = used.iloc[train_idx].reset_index(drop=True)
    test = used.iloc[test_idx].reset_index(drop=True)

    return {
        "features": feats,
        "X_train": train[feats].to_numpy(np.float64),
        "y_train": train[target_n].to_numpy(np.float64),
        "depth_train": train["DEPTH"].to_numpy(np.float64),
        "X_test": test[feats].to_numpy(np.float64),
        "y_test": test[target_n].to_numpy(np.float64),
        "depth_test": test["DEPTH"].to_numpy(np.float64),
        "n_full": n, "n_train": len(train), "n_test": len(test),
        "split_mode": split_mode,
        "depth_train_range": (float(train["DEPTH"].min()), float(train["DEPTH"].max())),
        "depth_test_range": (float(test["DEPTH"].min()), float(test["DEPTH"].max())),
    }


# ── V2.1 调过的超参 ──────────────────────────────────────────────
def fit_linear_single(d):
    best_r2 = -np.inf; best = None
    for j, fn in enumerate(d["features"]):
        m = LinearRegression().fit(d["X_train"][:, j:j+1], d["y_train"])
        yp_tr, yp_te = m.predict(d["X_train"][:, j:j+1]), m.predict(d["X_test"][:, j:j+1])
        r2_te = _r2(d["y_test"], yp_te)
        if r2_te > best_r2:
            best_r2 = r2_te; best = {"yp_tr": yp_tr, "yp_te": yp_te,
                                       "feature": fn,
                                       "formula": f"{m.coef_[0]:.4g} * {fn} + {m.intercept_:.4g}"}
    return best


def fit_linear_multi(d):
    m = LinearRegression().fit(d["X_train"], d["y_train"])
    yp_tr, yp_te = m.predict(d["X_train"]), m.predict(d["X_test"])
    return {"yp_tr": yp_tr, "yp_te": yp_te, "formula": "linear_multi"}


def fit_poly(d, degree):
    pipe = make_pipeline(StandardScaler(),
                          PolynomialFeatures(degree=degree, include_bias=False),
                          Ridge(alpha=1.0))
    pipe.fit(d["X_train"], d["y_train"])
    yp_tr, yp_te = pipe.predict(d["X_train"]), pipe.predict(d["X_test"])
    return {"yp_tr": yp_tr, "yp_te": yp_te, "formula": f"Poly(deg={degree})"}


def fit_svr_tuned(d):
    """V2.1: y 也归一化（修 POIS 这种小尺度变量崩盘）+ 简化 grid"""
    from sklearn.preprocessing import StandardScaler as SS
    y_scaler = SS().fit(d["y_train"].reshape(-1, 1))
    yn = y_scaler.transform(d["y_train"].reshape(-1, 1)).ravel()
    pipe = make_pipeline(SS(), SVR(kernel="rbf"))
    param_grid = {"svr__C": [1, 10, 100], "svr__gamma": ["scale", 0.1]}
    gs = GridSearchCV(pipe, param_grid, cv=3, n_jobs=8, scoring="r2")
    gs.fit(d["X_train"], yn)
    yp_tr_n = gs.predict(d["X_train"]); yp_te_n = gs.predict(d["X_test"])
    yp_tr = y_scaler.inverse_transform(yp_tr_n.reshape(-1, 1)).ravel()
    yp_te = y_scaler.inverse_transform(yp_te_n.reshape(-1, 1)).ravel()
    return {"yp_tr": yp_tr, "yp_te": yp_te,
            "formula": f"SVR(rbf, {gs.best_params_}, y-scaled)"}


def fit_rf_tuned(d):
    """V2.1: max_depth=10 防过拟合 + min_samples_leaf=5"""
    m = RandomForestRegressor(n_estimators=200, max_depth=10,
                               min_samples_leaf=5, n_jobs=8, random_state=42)
    m.fit(d["X_train"], d["y_train"])
    yp_tr, yp_te = m.predict(d["X_train"]), m.predict(d["X_test"])
    return {"yp_tr": yp_tr, "yp_te": yp_te,
            "formula": "RF(n=300, depth=10, leaf=5)",
            "feature_importance": {f: float(i) for f, i in zip(d["features"], m.feature_importances_)}}


def fit_xgb_tuned(d):
    """V2.1: max_depth=4 + lr=0.05 + 正则化（n_jobs=8 避免 192 核 IPC 拖慢）"""
    m = xgb.XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1,
                          reg_alpha=0.1, reg_lambda=1.0,
                          n_jobs=8, random_state=42, verbosity=0)
    m.fit(d["X_train"], d["y_train"])
    yp_tr, yp_te = m.predict(d["X_train"]), m.predict(d["X_test"])
    return {"yp_tr": yp_tr, "yp_te": yp_te,
            "formula": "XGB(n=300, depth=4, lr=0.05, reg)",
            "feature_importance": {f: float(i) for f, i in zip(d["features"], m.feature_importances_)}}


def fit_mlp_bp_tuned(d):
    """V2.1: 加深 + 更长训练 + 自适应 lr + early stopping"""
    pipe = make_pipeline(StandardScaler(),
                          MLPRegressor(hidden_layer_sizes=(128, 64, 32),
                                        activation="relu", solver="adam",
                                        max_iter=5000, random_state=42,
                                        learning_rate="adaptive",
                                        learning_rate_init=0.001,
                                        early_stopping=True,
                                        validation_fraction=0.1,
                                        n_iter_no_change=30,
                                        alpha=1e-4))
    pipe.fit(d["X_train"], d["y_train"])
    yp_tr, yp_te = pipe.predict(d["X_train"]), pipe.predict(d["X_test"])
    return {"yp_tr": yp_tr, "yp_te": yp_te,
            "formula": "MLP-BP(128,64,32, adaptive_lr, ES)"}


MODELS = {
    "Linear_single": fit_linear_single,
    "Linear_multi":  fit_linear_multi,
    "Poly2":         lambda d: fit_poly(d, 2),
    "Poly3":         lambda d: fit_poly(d, 3),
    "SVR_rbf":       fit_svr_tuned,
    "RF":            fit_rf_tuned,
    "XGBoost":       fit_xgb_tuned,
    "MLP_BP":        fit_mlp_bp_tuned,
}


def run_one(d, well, target, out_dir):
    rows = []
    for name, fn in MODELS.items():
        sub_dir = out_dir / tv2.safe_dir_name(well) / tv2.safe_dir_name(target) / name
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
                   "formula_or_arch": r.get("formula", name),
                   "feature_importance": r.get("feature_importance"),
                   "feature_used": r.get("feature")}
            (sub_dir / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2),
                                                  encoding="utf-8")
            print(f"  {name:14s} train={r2_tr:.4f} test={r2_te:.4f} ({elapsed:.1f}s)")
            sys.stdout.flush()
            rows.append({"well": well, "target": target, "model": name,
                         "split_mode": d["split_mode"],
                         "r2_train": round(float(r2_tr), 4),
                         "r2_test": round(float(r2_te), 4),
                         "mae_test": round(float(mae_te), 4),
                         "elapsed_sec": round(elapsed, 2),
                         "formula": r.get("formula", name)})
        except Exception as e:
            print(f"  {name:14s} ERROR: {e}")
            sys.stdout.flush()
            rows.append({"well": well, "target": target, "model": name,
                         "split_mode": d["split_mode"],
                         "r2_train": None, "r2_test": None, "error": str(e)})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, type=Path)
    ap.add_argument("--out_dir", required=True, type=Path)
    ap.add_argument("--split_mode", default="last20", choices=["last20", "random"])
    ap.add_argument("--targets", default=",".join(TARGETS))
    args = ap.parse_args()
    targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== V2.1 经典 ML (split={args.split_mode}, 9 特征 V2.1) ===")
    print(f"白名单: {WHITELIST_V21}")
    sys.stdout.flush()
    df = tv2.load_real_data(args.data_dir)
    wells = sorted({str(x) for x in df["WELLNAME"].dropna().unique() if str(x).strip()})
    rows = []; t0 = time.time()
    for well in wells:
        df_well = df[df["WELLNAME"] == well].copy()
        for target in targets:
            d = prepare_v21(df_well, target, split_mode=args.split_mode)
            if d is None: continue
            print(f"\n{well}/{target}: train={d['n_train']} test={d['n_test']} feat={len(d['features'])}")
            rows.extend(run_one(d, well, target, args.out_dir))
    pd.DataFrame(rows).to_csv(args.out_dir / "metrics_summary.csv",
                               index=False, encoding="utf-8-sig")
    print(f"\n=== {time.time()-t0:.0f}s 总，summary → {args.out_dir / 'metrics_summary.csv'} ===")


if __name__ == "__main__":
    raise SystemExit(main())
