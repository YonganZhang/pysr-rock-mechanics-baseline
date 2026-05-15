"""
Poly2/Poly3 robust 版本 — 修 OOD 数值发散
4 个修复技巧：
1. Ridge alpha 大幅增强（1 → 1000）
2. 预测剪裁到训练集 y range ±50%（防 10¹¹ 量级爆炸）
3. interaction_only=True 在 Poly3（特征数从 219 降到 ~36）
4. 特征 clipping：测试集特征 standardized 后 clip 到 [-3, 3] 范围
   （阻止外推特征值远离训练集分布）

输出到 真跑结果-baseline-v2/经典方法/真实数据组-ZXX/train_results/{Well}/{Target}/Poly{2,3}_robust/
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import train_v2 as tv2
import train_v21_classical as tv21

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


def _r2(y, yp):
    ss = float(np.sum((y - yp) ** 2))
    tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss / max(tot, 1e-12)


class RobustPoly:
    """改进版多项式回归：强正则化 + 特征/预测双重剪裁"""
    def __init__(self, degree=2, alpha=1000.0, interaction_only=False,
                 x_clip=3.0, y_pad=0.5):
        self.degree = degree
        self.alpha = alpha
        self.interaction_only = interaction_only
        self.x_clip = x_clip          # 标准化后 x 剪裁到 ±x_clip σ
        self.y_pad = y_pad            # 预测剪裁到 [y_min - y_pad*range, y_max + y_pad*range]
        self.x_scaler = None
        self.poly = None
        self.ridge = None
        self.y_min = None
        self.y_max = None

    def fit(self, X, y):
        self.x_scaler = StandardScaler()
        Xs = self.x_scaler.fit_transform(X)
        # 训练时不 clip（让模型见过真实范围）
        self.poly = PolynomialFeatures(degree=self.degree, include_bias=False,
                                         interaction_only=self.interaction_only)
        Xp = self.poly.fit_transform(Xs)
        self.ridge = Ridge(alpha=self.alpha)
        self.ridge.fit(Xp, y)
        # 记录 y 训练范围
        self.y_min = float(y.min())
        self.y_max = float(y.max())
        return self

    def predict(self, X):
        Xs = self.x_scaler.transform(X)
        # 测试时 clip x 到 ±3σ（防外推特征值导致高阶项爆炸）
        Xs_clip = np.clip(Xs, -self.x_clip, self.x_clip)
        Xp = self.poly.transform(Xs_clip)
        yp = self.ridge.predict(Xp)
        # 预测剪裁到训练集 y range ± pad
        rng = self.y_max - self.y_min
        lo = self.y_min - self.y_pad * rng
        hi = self.y_max + self.y_pad * rng
        return np.clip(yp, lo, hi)


def fit_poly_robust(d, degree, alpha=1000.0, interaction_only=False):
    m = RobustPoly(degree=degree, alpha=alpha, interaction_only=interaction_only,
                    x_clip=3.0, y_pad=0.5)
    m.fit(d["X_train"], d["y_train"])
    return {"yp_tr": m.predict(d["X_train"]),
            "yp_te": m.predict(d["X_test"]),
            "formula": f"Poly(deg={degree}, alpha={alpha}, x_clip=3σ, y_pad=0.5, interaction_only={interaction_only})"}


CONFIGS = {
    "Poly2_robust": {"degree": 2, "alpha": 1000.0, "interaction_only": False},
    "Poly3_robust": {"degree": 3, "alpha": 1000.0, "interaction_only": True},  # 高阶必须 interaction_only
}


def run_one(d, well, target, out_dir):
    rows = []
    for name, cfg in CONFIGS.items():
        sub_dir = out_dir / tv2.safe_dir_name(well) / tv2.safe_dir_name(target) / name
        if (sub_dir / "config.json").exists():
            print(f"  {name:14s} SKIP (已存在)")
            continue
        sub_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        try:
            r = fit_poly_robust(d, **cfg)
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
            out_cfg = {"well": well, "target": target, "model": name,
                       "split_mode": d["split_mode"],
                       "n_train": d["n_train"], "n_test": d["n_test"],
                       "elapsed_sec": round(elapsed, 2),
                       "r2_train": round(float(r2_tr), 4),
                       "r2_test": round(float(r2_te), 4),
                       "mae_test": round(float(mae_te), 4),
                       "formula_or_arch": r["formula"]}
            (sub_dir / "config.json").write_text(json.dumps(out_cfg, ensure_ascii=False, indent=2),
                                                  encoding="utf-8")
            print(f"  {name:14s} train={r2_tr:.4f} test={r2_te:.4f} ({elapsed:.1f}s)")
            sys.stdout.flush()
            rows.append({"well": well, "target": target, "model": name,
                         "r2_train": round(float(r2_tr), 4),
                         "r2_test": round(float(r2_te), 4),
                         "elapsed_sec": round(elapsed, 2)})
        except Exception as e:
            print(f"  {name:14s} ERROR: {e}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, type=Path)
    ap.add_argument("--out_dir", required=True, type=Path)
    ap.add_argument("--split_mode", default="last20", choices=["last20", "random"])
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Poly Robust (Ridge α=1000 + x clip ±3σ + y pad 0.5) split={args.split_mode} ===")
    sys.stdout.flush()
    df = tv2.load_real_data(args.data_dir)
    wells = sorted({str(x) for x in df["WELLNAME"].dropna().unique() if str(x).strip()})
    rows = []; t0 = time.time()
    for well in wells:
        df_well = df[df["WELLNAME"] == well].copy()
        for target in tv2.TARGETS:
            d = tv21.prepare_v21(df_well, target, split_mode=args.split_mode)
            if d is None: continue
            print(f"\n{well}/{target}: train={d['n_train']} test={d['n_test']}")
            sys.stdout.flush()
            rows.extend(run_one(d, well, target, args.out_dir))
    pd.DataFrame(rows).to_csv(args.out_dir / "metrics_summary_poly_robust.csv",
                               index=False, encoding="utf-8-sig")
    print(f"\n=== {time.time()-t0:.0f}s 完成 ===")


if __name__ == "__main__":
    raise SystemExit(main())
