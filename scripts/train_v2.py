"""
V2 真跑 baseline (修正 V1 三大 bug)

V1 → V2 改动：
  ❌ V1 用了 BULK 当输入（弹性参数泄漏）→ ✅ V2 仅 10 条原始测井白名单
  ❌ V1 PySR/KAN 训练集不对称 → ✅ V2 共享同一 train/test split
  ❌ V1 用全量训全量测 → ✅ V2 按深度排序后 last 20% 连续段做 OOD test
  ❌ V1 包含浅部 measured=0 填充 → ✅ V2 过滤 measured.abs>1e-9 + 10 特征全有效

模型：
  PySR: niter=80, populations=20, maxsize=30, ops=[+,-,*,/]
  KAN: width=[10, 10, 1], grid=5, k=3, LBFGS 50 步

输出：
  out_dir/{Well}/{Target}/
    config.json      — 训练 R² + 测试 R² + 公式（PySR）或 mae（KAN）
    predictions.csv  — depth, measured, predicted, split (train/test)
"""
from __future__ import annotations
import argparse
import json
import sys
import time
import warnings
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ZHY_TRAE = Path(__file__).resolve().parents[2] / "学生-张涵宇-PYSR改进" / ".trae" / "skills" / "pysr_equation_fit"
sys.path.insert(0, str(ZHY_TRAE))
from a import build_dataset_from_dir, _normalize_column_name, _r2_score
from helpers import safe_dir_name

# ── V2 严格白名单：仅原始仪器测井（V2.1: AC 因质量差移除） ─────────
WHITELIST = ["DTC", "DTS", "DEN", "GR", "CNL", "LLD", "LLS", "RMSC", "CAL"]
TARGETS = ["YMOD", "SMOD", "POIS", "SHMAX"]

# ── 黑名单：所有衍生 / 解释 / 弹性参数 / 矿物 / 流体 / 孔隙 ─────────
# 用作健壮性检查（保证白名单里没有意外混入衍生量）
DERIVED_BLACKLIST = {
    # 弹性力学（目标本身 + 同源）
    "YMOD", "SMOD", "BMOD", "BULK", "POIS", "SHMAX", "SHMIN",
    "BRIT", "BRITT", "BRITL", "BRITS",
    # 流体 / 孔隙 / 渗透
    "FI", "PF", "Pf", "PV", "JW", "YLCXS", "TOC", "GAS", "GAS2",
    "POR", "POR2", "PORF", "PORT", "PORW", "PHIE", "PHIT",
    "PHIE_T1T2", "PHIT_T1T2", "SW", "SWE_T1T2", "SWT_T1T2",
    "SO", "PERM", "KSDR_NMR", "KTIM_NMR", "KTIM_TAPER_NMR",
    "MRP_CMR", "T2CUTOFF_CMR", "T2LM_DI_CMR",
    "FFV_3MS_CMR", "FFV_CMR", "FFV_T1T2", "FFV_10MS_CMR",
    # 矿物 / 元素含量
    "QT", "QA", "Qa2", "VF", "Vf2", "K",
    # 自然伽马能谱衍生组合
    "KTH", "TH", "U",
}

WELLNAME_FIX = {"足212（导眼井）": "ZXX2"}
WELLNAME_NORMALIZE = {"ZXX3井": "ZXX3", "ZXX7井": "ZXX7"}
META_COLS = {"SOURCE_FILE", "WELLNAME", "DEPTH"}

PYSR_NITER = 80
PYSR_POPULATIONS = 20
PYSR_MAXSIZE = 30
PYSR_BIN_OPS = ["+", "-", "*", "/"]

KAN_GRID = 5
KAN_K = 3
KAN_HIDDEN = [10]
KAN_STEPS = 50
KAN_LR = 0.01

TEST_FRAC = 0.20  # 末 20% 深度做 OOD test


def load_real_data(data_dir: Path) -> pd.DataFrame:
    df = build_dataset_from_dir(data_dir, keep_common_columns_only=False)
    if "WELLNAME" in df.columns:
        for k, v in WELLNAME_FIX.items():
            df.loc[df["WELLNAME"] == k, "WELLNAME"] = v
        for k, v in WELLNAME_NORMALIZE.items():
            df.loc[df["WELLNAME"] == k, "WELLNAME"] = v
    # 健壮性：确认白名单里没有黑名单
    leaks = [c for c in WHITELIST if c in DERIVED_BLACKLIST]
    if leaks:
        raise RuntimeError(f"白名单里有衍生量泄漏: {leaks}")
    return df


def prepare_well_target(df_well: pd.DataFrame, target: str,
                          split_mode: str = "last20", random_seed: int = 42) -> Optional[dict]:
    """返回 {X_train, y_train, X_test, y_test, depth_train, depth_test, features}"""
    target_n = _normalize_column_name(target)
    if target_n not in df_well.columns:
        return None
    feats = [c for c in WHITELIST if c in df_well.columns]
    if len(feats) < 4:
        return None

    cols = ["DEPTH", target_n] + feats
    used = df_well[cols].copy()
    for c in cols:
        used[c] = pd.to_numeric(used[c], errors="coerce")
    used = used.replace([np.inf, -np.inf], np.nan).dropna()
    used = used[used[target_n].abs() > 1e-9].reset_index(drop=True)
    if len(used) < 100:
        return None

    used = used.sort_values("DEPTH").reset_index(drop=True)
    n = len(used)
    n_test = max(int(round(n * TEST_FRAC)), 30)
    n_train = n - n_test

    if split_mode == "last20":
        train = used.iloc[:n_train].reset_index(drop=True)
        test = used.iloc[n_train:].reset_index(drop=True)
    elif split_mode == "random":
        rng = np.random.RandomState(random_seed)
        idx = np.arange(n); rng.shuffle(idx)
        train_idx = sorted(idx[:n_train]); test_idx = sorted(idx[n_train:])
        train = used.iloc[train_idx].reset_index(drop=True)
        test = used.iloc[test_idx].reset_index(drop=True)
    else:
        raise ValueError(split_mode)

    return {
        "features": feats,
        "X_train": train[feats].to_numpy(np.float64),
        "y_train": train[target_n].to_numpy(np.float64),
        "depth_train": train["DEPTH"].to_numpy(np.float64),
        "X_test": test[feats].to_numpy(np.float64),
        "y_test": test[target_n].to_numpy(np.float64),
        "depth_test": test["DEPTH"].to_numpy(np.float64),
        "n_full": n, "n_train": n_train, "n_test": n_test,
        "depth_train_range": (float(train["DEPTH"].min()), float(train["DEPTH"].max())),
        "depth_test_range": (float(test["DEPTH"].min()), float(test["DEPTH"].max())),
    }


def fit_pysr(d: dict, sub_dir: Path, well: str, target: str) -> dict:
    from pysr import PySRRegressor
    sub_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    m = PySRRegressor(
        niterations=PYSR_NITER, populations=PYSR_POPULATIONS, maxsize=PYSR_MAXSIZE,
        binary_operators=PYSR_BIN_OPS, unary_operators=[],
        random_state=42, deterministic=False, verbosity=0,
        tempdir=str(sub_dir / "_pysr_tmp"), output_directory=str(sub_dir / "_pysr_tmp"),
        progress=False,
    )
    m.fit(d["X_train"], d["y_train"], variable_names=d["features"])
    eq = m.get_best()
    yp_tr = m.predict(d["X_train"])
    yp_te = m.predict(d["X_test"])
    elapsed = time.time() - t0
    r2_tr = _r2_score(d["y_train"], yp_tr)
    r2_te = _r2_score(d["y_test"], yp_te)
    mae_tr = float(np.mean(np.abs(d["y_train"] - yp_tr)))
    mae_te = float(np.mean(np.abs(d["y_test"] - yp_te)))
    formula = str(eq["equation"]) if hasattr(eq, "get") else str(m.sympy())
    complexity = int(eq.get("complexity", -1)) if hasattr(eq, "get") else -1

    pred_df = pd.DataFrame({
        "depth": np.concatenate([d["depth_train"], d["depth_test"]]),
        "measured": np.concatenate([d["y_train"], d["y_test"]]),
        "predicted": np.concatenate([yp_tr, yp_te]),
        "split": ["train"] * d["n_train"] + ["test"] * d["n_test"],
    }).sort_values("depth").reset_index(drop=True)
    pred_df.to_csv(sub_dir / "predictions.csv", index=False, encoding="utf-8-sig")

    cfg = {
        "well": well, "target": target, "model": "PySR",
        "n_full": d["n_full"], "n_train": d["n_train"], "n_test": d["n_test"],
        "depth_train_range": d["depth_train_range"],
        "depth_test_range": d["depth_test_range"],
        "features": d["features"], "n_features": len(d["features"]),
        "pysr_niter": PYSR_NITER, "pysr_populations": PYSR_POPULATIONS,
        "pysr_maxsize": PYSR_MAXSIZE,
        "elapsed_sec": round(elapsed, 1),
        "r2_train": round(float(r2_tr), 4), "r2_test": round(float(r2_te), 4),
        "mae_train": round(float(mae_tr), 4), "mae_test": round(float(mae_te), 4),
        "complexity": complexity, "formula": formula,
    }
    (sub_dir / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2),
                                          encoding="utf-8")
    return cfg


def fit_kan(d: dict, sub_dir: Path, well: str, target: str) -> dict:
    import torch
    from kan import KAN
    sub_dir.mkdir(parents=True, exist_ok=True)

    X_tr_mu, X_tr_sigma = d["X_train"].mean(0), d["X_train"].std(0) + 1e-9
    y_tr_mu, y_tr_sigma = d["y_train"].mean(), d["y_train"].std() + 1e-9
    Xn_tr = (d["X_train"] - X_tr_mu) / X_tr_sigma
    Xn_te = (d["X_test"] - X_tr_mu) / X_tr_sigma  # 用训练集统计量
    yn_tr = (d["y_train"] - y_tr_mu) / y_tr_sigma

    torch.manual_seed(42)
    np.random.seed(42)
    width = [len(d["features"])] + KAN_HIDDEN + [1]
    model = KAN(width=width, grid=KAN_GRID, k=KAN_K, seed=42, device="cpu")
    Xt_tr = torch.tensor(Xn_tr, dtype=torch.float32)
    yt_tr = torch.tensor(yn_tr.reshape(-1, 1), dtype=torch.float32)
    Xt_te = torch.tensor(Xn_te, dtype=torch.float32)
    yt_te = torch.tensor(((d["y_test"] - y_tr_mu) / y_tr_sigma).reshape(-1, 1),
                         dtype=torch.float32)

    dataset = {"train_input": Xt_tr, "train_label": yt_tr,
               "test_input": Xt_te, "test_label": yt_te}
    t0 = time.time()
    model.fit(dataset, opt="LBFGS", steps=KAN_STEPS, lr=KAN_LR, lamb=0.0)
    elapsed = time.time() - t0

    with torch.no_grad():
        yp_tr_n = model(Xt_tr).cpu().numpy().reshape(-1)
        yp_te_n = model(Xt_te).cpu().numpy().reshape(-1)
    yp_tr = yp_tr_n * y_tr_sigma + y_tr_mu
    yp_te = yp_te_n * y_tr_sigma + y_tr_mu
    r2_tr = _r2_score(d["y_train"], yp_tr)
    r2_te = _r2_score(d["y_test"], yp_te)
    mae_tr = float(np.mean(np.abs(d["y_train"] - yp_tr)))
    mae_te = float(np.mean(np.abs(d["y_test"] - yp_te)))

    pred_df = pd.DataFrame({
        "depth": np.concatenate([d["depth_train"], d["depth_test"]]),
        "measured": np.concatenate([d["y_train"], d["y_test"]]),
        "predicted": np.concatenate([yp_tr, yp_te]),
        "split": ["train"] * d["n_train"] + ["test"] * d["n_test"],
    }).sort_values("depth").reset_index(drop=True)
    pred_df.to_csv(sub_dir / "predictions.csv", index=False, encoding="utf-8-sig")

    cfg = {
        "well": well, "target": target, "model": "KAN_standard",
        "n_full": d["n_full"], "n_train": d["n_train"], "n_test": d["n_test"],
        "depth_train_range": d["depth_train_range"],
        "depth_test_range": d["depth_test_range"],
        "features": d["features"], "n_features": len(d["features"]),
        "kan_width": width, "kan_grid": KAN_GRID, "kan_k": KAN_K,
        "kan_steps": KAN_STEPS, "kan_lr": KAN_LR,
        "elapsed_sec": round(elapsed, 1),
        "r2_train": round(float(r2_tr), 4), "r2_test": round(float(r2_te), 4),
        "mae_train": round(float(mae_tr), 4), "mae_test": round(float(mae_te), 4),
        "y_mean": float(y_tr_mu), "y_std": float(y_tr_sigma),
    }
    (sub_dir / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2),
                                          encoding="utf-8")
    try:
        torch.save(model.state_dict(), sub_dir / "kan_state.pt")
    except Exception:
        pass
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, type=Path)
    ap.add_argument("--out_pysr", required=True, type=Path)
    ap.add_argument("--out_kan", required=True, type=Path)
    ap.add_argument("--targets", default=",".join(TARGETS))
    ap.add_argument("--split_mode", default="last20", choices=["last20", "random"])
    ap.add_argument("--skip_pysr", action="store_true")
    ap.add_argument("--skip_kan", action="store_true")
    args = ap.parse_args()

    targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    args.out_pysr.mkdir(parents=True, exist_ok=True)
    args.out_kan.mkdir(parents=True, exist_ok=True)

    print(f"=== V2 真跑 (严格白名单 + 末 20% 连续 OOD test) ===")
    print(f"白名单: {WHITELIST}")
    print(f"目标:   {targets}")
    print(f"data_dir: {args.data_dir}")
    sys.stdout.flush()

    df = load_real_data(args.data_dir)
    wells = sorted({str(x) for x in df["WELLNAME"].dropna().unique() if str(x).strip()})
    print(f"wells: {wells}")
    sys.stdout.flush()

    rows_pysr, rows_kan = [], []
    t0 = time.time()
    for well in wells:
        df_well = df[df["WELLNAME"] == well].copy()
        for target in targets:
            d = prepare_well_target(df_well, target, split_mode=args.split_mode)
            if d is None:
                print(f"  SKIP {well}/{target}")
                continue
            print(f"\n{well}/{target}: n={d['n_full']} (train {d['n_train']} / test {d['n_test']}), "
                  f"feat={len(d['features'])}, "
                  f"depth_test={d['depth_test_range'][0]:.0f}-{d['depth_test_range'][1]:.0f}m")
            sys.stdout.flush()

            if not args.skip_pysr:
                pysr_dir = args.out_pysr / safe_dir_name(well) / safe_dir_name(target)
                if (pysr_dir / "config.json").exists():
                    print(f"  PySR SKIP (已存在) {well}/{target}")
                    sys.stdout.flush()
                    continue
                try:
                    cfg = fit_pysr(d, pysr_dir, well, target)
                    print(f"  PySR train R²={cfg['r2_train']:.4f} | test R²={cfg['r2_test']:.4f} "
                          f"| {cfg['elapsed_sec']:.0f}s")
                    rows_pysr.append(cfg)
                except Exception as e:
                    print(f"  PySR ERROR: {e}")
                sys.stdout.flush()

            if not args.skip_kan:
                try:
                    cfg = fit_kan(d, args.out_kan / safe_dir_name(well) / safe_dir_name(target),
                                  well, target)
                    print(f"  KAN  train R²={cfg['r2_train']:.4f} | test R²={cfg['r2_test']:.4f} "
                          f"| {cfg['elapsed_sec']:.0f}s")
                    rows_kan.append(cfg)
                except Exception as e:
                    print(f"  KAN ERROR: {e}")
                sys.stdout.flush()

    if rows_pysr:
        df_p = pd.DataFrame([{k: v for k, v in r.items() if not isinstance(v, (list, tuple))}
                             for r in rows_pysr])
        df_p.to_csv(args.out_pysr / "metrics_summary.csv",
                    index=False, encoding="utf-8-sig")
    if rows_kan:
        df_k = pd.DataFrame([{k: v for k, v in r.items() if not isinstance(v, (list, tuple))}
                             for r in rows_kan])
        df_k.to_csv(args.out_kan / "metrics_summary.csv",
                    index=False, encoding="utf-8-sig")

    print(f"\n=== 总耗时 {time.time()-t0:.0f}s ===")


if __name__ == "__main__":
    raise SystemExit(main())
