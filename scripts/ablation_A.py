"""
消融 A: leave-one-feature-out — 9 个特征逐一移除看 R² 退化
对每口井每个目标，用 9 - 1 = 8 个特征训练，看哪个特征不可少。

模型：Linear_single, Linear_multi, RF, XGBoost, MLP_BP, gplearn
（PySR 太慢，作为 baseline 只跑 1 次完整 9 特征作对照）

输出: 真跑结果-baseline-v2/消融/A_特征消融/{model}/{removed}/{Well}/{Target}/{config.json,pred.csv}
+ summary table 9 特征 × 模型 = R² 退化矩阵
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

WHITELIST = tv2.WHITELIST  # 9 个
TARGETS = tv2.TARGETS

# 跑哪些模型
# A 砍掉 gplearn (90s/fit × 160 = 4h 拖太久)，保留 5 个快模型
ABLATION_MODELS = {
    "Linear_single": tv21.fit_linear_single,
    "Linear_multi":  tv21.fit_linear_multi,
    "RF":            tv21.fit_rf_tuned,
    "XGBoost":       tv21.fit_xgb_tuned,
    "MLP_BP":        tv21.fit_mlp_bp_tuned,
}


def _r2(y, yp):
    ss = float(np.sum((y - yp) ** 2))
    tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss / max(tot, 1e-12)


def prepare_with_features(df_well: pd.DataFrame, target: str, feats: list,
                            split_mode: str = "last20"):
    """跟 prepare_v21 一样但 features 由调用方指定"""
    target_n = tv2._normalize_column_name(target)
    if target_n not in df_well.columns: return None
    feats_avail = [c for c in feats if c in df_well.columns]
    if len(feats_avail) < 1: return None
    cols = ["DEPTH", target_n] + feats_avail
    used = df_well[cols].copy()
    for c in cols:
        used[c] = pd.to_numeric(used[c], errors="coerce")
    used = used.replace([np.inf, -np.inf], np.nan).dropna()
    used = used[used[target_n].abs() > 1e-9].reset_index(drop=True)
    if len(used) < 100: return None
    used = used.sort_values("DEPTH").reset_index(drop=True)
    n = len(used); n_test = max(int(round(n * 0.2)), 30); n_train = n - n_test
    train = used.iloc[:n_train]; test = used.iloc[n_train:]
    return {
        "features": feats_avail,
        "X_train": train[feats_avail].to_numpy(np.float64),
        "y_train": train[target_n].to_numpy(np.float64),
        "depth_train": train["DEPTH"].to_numpy(np.float64),
        "X_test": test[feats_avail].to_numpy(np.float64),
        "y_test": test[target_n].to_numpy(np.float64),
        "depth_test": test["DEPTH"].to_numpy(np.float64),
        "n_full": n, "n_train": len(train), "n_test": len(test),
        "split_mode": split_mode,
        "depth_train_range": (float(train["DEPTH"].min()), float(train["DEPTH"].max())),
        "depth_test_range": (float(test["DEPTH"].min()), float(test["DEPTH"].max())),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, type=Path)
    ap.add_argument("--out_dir", required=True, type=Path)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== 消融 A: leave-one-feature-out ===")
    print(f"完整 9 特征: {WHITELIST}")
    sys.stdout.flush()

    df = tv2.load_real_data(args.data_dir)
    wells = sorted({str(x) for x in df["WELLNAME"].dropna().unique() if str(x).strip()})

    rows = []; t0 = time.time()
    # 完整 9 特征作 baseline (removed="(none)")
    feature_settings = [("(none)", WHITELIST)] + [
        (rm, [c for c in WHITELIST if c != rm]) for rm in WHITELIST
    ]
    total_settings = len(feature_settings) * len(wells) * len(TARGETS) * len(ABLATION_MODELS)
    print(f"总实验数: {total_settings}")
    sys.stdout.flush()
    done = 0

    for removed, feats in feature_settings:
        print(f"\n--- 移除特征: {removed} (剩 {len(feats)} 个) ---")
        sys.stdout.flush()
        for well in wells:
            df_well = df[df["WELLNAME"] == well].copy()
            for target in TARGETS:
                d = prepare_with_features(df_well, target, feats)
                if d is None: continue
                for mname, mfn in ABLATION_MODELS.items():
                    sub_dir = args.out_dir / mname / removed / tv2.safe_dir_name(well) / tv2.safe_dir_name(target)
                    if (sub_dir / "config.json").exists():
                        done += 1; continue
                    sub_dir.mkdir(parents=True, exist_ok=True)
                    t1 = time.time()
                    try:
                        r = mfn(d)
                        r2_tr = _r2(d["y_train"], r["yp_tr"])
                        r2_te = _r2(d["y_test"], r["yp_te"])
                        cfg = {"well": well, "target": target, "removed_feature": removed,
                               "model": mname, "n_features": len(feats),
                               "features": feats,
                               "r2_train": round(float(r2_tr), 4),
                               "r2_test": round(float(r2_te), 4),
                               "elapsed_sec": round(time.time()-t1, 1),
                               "formula": r.get("formula", mname)[:200]}
                        (sub_dir / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2),
                                                              encoding="utf-8")
                        rows.append(cfg)
                    except Exception as e:
                        rows.append({"well": well, "target": target, "removed_feature": removed,
                                     "model": mname, "error": str(e)})
                    done += 1
        print(f"  进度 {done}/{total_settings} ({100*done/total_settings:.0f}%)")
        sys.stdout.flush()

    pd.DataFrame(rows).to_csv(args.out_dir / "ablation_A_summary.csv",
                               index=False, encoding="utf-8-sig")
    print(f"\n=== {time.time()-t0:.0f}s 总，summary → ablation_A_summary.csv ===")


if __name__ == "__main__":
    raise SystemExit(main())
