"""
消融 B: split 比例敏感性 — 看 OOD 测试比例从 10% 到 40% 的 R² 变化

跑：60/40, 70/30, 80/20 (V2.1 baseline), 90/10
模型：Linear_single, RF, XGBoost, MLP_BP, gplearn (PySR 太慢跳过)

每个 split 比例下取 last X% 作 OOD test。
输出 R² vs test_frac 曲线，量化"外推距离 → 模型崩坏速度"
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import train_v2 as tv2
import train_v21_classical as tv21
import train_other_sr as tosr

WHITELIST = tv2.WHITELIST
TARGETS = tv2.TARGETS

MODELS = {
    "Linear_single": tv21.fit_linear_single,
    "RF":            tv21.fit_rf_tuned,
    "XGBoost":       tv21.fit_xgb_tuned,
    "MLP_BP":        tv21.fit_mlp_bp_tuned,
}

TEST_FRACS = [0.10, 0.20, 0.30, 0.40]


def _r2(y, yp):
    ss = float(np.sum((y - yp) ** 2))
    tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss / max(tot, 1e-12)


def prepare(df_well, target, test_frac):
    target_n = tv2._normalize_column_name(target)
    if target_n not in df_well.columns: return None
    feats = [c for c in WHITELIST if c in df_well.columns]
    cols = ["DEPTH", target_n] + feats
    used = df_well[cols].copy()
    for c in cols:
        used[c] = pd.to_numeric(used[c], errors="coerce")
    used = used.replace([np.inf, -np.inf], np.nan).dropna()
    used = used[used[target_n].abs() > 1e-9].reset_index(drop=True)
    if len(used) < 100: return None
    used = used.sort_values("DEPTH").reset_index(drop=True)
    n = len(used); n_test = max(int(round(n * test_frac)), 30); n_train = n - n_test
    train = used.iloc[:n_train]; test = used.iloc[n_train:]
    return {
        "features": feats,
        "X_train": train[feats].to_numpy(np.float64),
        "y_train": train[target_n].to_numpy(np.float64),
        "depth_train": train["DEPTH"].to_numpy(np.float64),
        "X_test": test[feats].to_numpy(np.float64),
        "y_test": test[target_n].to_numpy(np.float64),
        "depth_test": test["DEPTH"].to_numpy(np.float64),
        "n_full": n, "n_train": len(train), "n_test": len(test),
        "split_mode": f"last{int(test_frac*100)}",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, type=Path)
    ap.add_argument("--out_dir", required=True, type=Path)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== 消融 B: split 比例敏感性 ===")
    print(f"test_frac: {TEST_FRACS}")
    sys.stdout.flush()

    df = tv2.load_real_data(args.data_dir)
    wells = sorted({str(x) for x in df["WELLNAME"].dropna().unique() if str(x).strip()})

    rows = []; t0 = time.time()
    for tf in TEST_FRACS:
        print(f"\n--- test_frac = {tf} (last {int(tf*100)}% by depth) ---")
        sys.stdout.flush()
        for well in wells:
            df_well = df[df["WELLNAME"] == well].copy()
            for target in TARGETS:
                d = prepare(df_well, target, tf)
                if d is None: continue
                for mname, mfn in MODELS.items():
                    sub_dir = args.out_dir / f"frac_{int(tf*100)}" / mname / tv2.safe_dir_name(well) / tv2.safe_dir_name(target)
                    if (sub_dir / "config.json").exists(): continue
                    sub_dir.mkdir(parents=True, exist_ok=True)
                    t1 = time.time()
                    try:
                        r = mfn(d)
                        r2_tr = _r2(d["y_train"], r["yp_tr"])
                        r2_te = _r2(d["y_test"], r["yp_te"])
                        cfg = {"well": well, "target": target, "test_frac": tf,
                               "model": mname, "n_train": d["n_train"], "n_test": d["n_test"],
                               "r2_train": round(float(r2_tr), 4),
                               "r2_test": round(float(r2_te), 4),
                               "elapsed_sec": round(time.time()-t1, 1)}
                        (sub_dir / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2),
                                                              encoding="utf-8")
                        rows.append(cfg)
                    except Exception as e:
                        rows.append({"well": well, "target": target, "test_frac": tf,
                                     "model": mname, "error": str(e)})

    pd.DataFrame(rows).to_csv(args.out_dir / "ablation_B_summary.csv",
                               index=False, encoding="utf-8-sig")
    print(f"\n=== {time.time()-t0:.0f}s, summary → ablation_B_summary.csv ===")


if __name__ == "__main__":
    raise SystemExit(main())
