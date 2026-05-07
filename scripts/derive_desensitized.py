"""
从真实数据组训练结果派生脱敏组（井名映射 + depth 偏移），不重跑训练。

映射规则：
  ZXX2 → WellA  depth offset +0       (论文 paper/data_desensitized/ 的脱敏方案保留 depth 量级)
  ZXX3 → WellB  depth offset +0
  ZXX6 → WellC  depth offset +0
  ZXX7 → WellD  depth offset +0

  (默认 depth 不偏移，保持脱敏前后数字易对照；如需扰动加 --depth_offset)

输入：<src>/{ZXX*}/{target}/{config.json,predictions.csv}
输出：<dst>/{Well*}/{target}/{config.json,predictions.csv} + metrics_summary.csv

调用：
  python derive_desensitized.py --src <PATH 真实组 train_results> --dst <PATH 脱敏组 train_results>
"""
from __future__ import annotations
import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

WELL_MAP = {"ZXX2": "WellA", "ZXX3": "WellB", "ZXX6": "WellC", "ZXX7": "WellD"}


def derive(src: Path, dst: Path, depth_offset: float = 0.0):
    dst.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    for old_well, new_well in WELL_MAP.items():
        src_well = src / old_well
        if not src_well.is_dir():
            print(f"  [SKIP] {old_well} 源目录不存在")
            continue
        for tgt_dir in sorted(src_well.iterdir()):
            if not tgt_dir.is_dir():
                continue
            target = tgt_dir.name
            new_dir = dst / new_well / target
            new_dir.mkdir(parents=True, exist_ok=True)

            # 处理 predictions.csv
            pred_src = tgt_dir / "predictions.csv"
            if pred_src.exists():
                df = pd.read_csv(pred_src)
                if depth_offset:
                    df["depth"] = df["depth"] + depth_offset
                df.to_csv(new_dir / "predictions.csv", index=False, encoding="utf-8-sig")

            # 处理 config.json
            cfg_src = tgt_dir / "config.json"
            if cfg_src.exists():
                cfg = json.loads(cfg_src.read_text(encoding="utf-8"))
                cfg["well"] = new_well
                cfg["original_well"] = old_well  # 留个映射对照（脱敏组内部，不外传）
                cfg["depth_offset_applied"] = depth_offset
                (new_dir / "config.json").write_text(
                    json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

                if cfg.get("status", "ok") != "skipped":
                    summary_rows.append({
                        "well": new_well, "target": target,
                        "status": "ok",
                        "pysr_r2": cfg.get("pysr_r2") or cfg.get("r2"),
                        "pysr_mae": cfg.get("pysr_mae") or cfg.get("mae"),
                        "poly_r2": cfg.get("poly_r2"),
                        "n_rows_used": cfg.get("n_rows_used"),
                        "n_features": cfg.get("n_features"),
                        "pysr_complexity": cfg.get("pysr_complexity"),
                        "pysr_formula": cfg.get("pysr_formula"),
                    })
            print(f"  {old_well}/{target} → {new_well}/{target}")

    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(dst / "metrics_summary.csv",
                                          index=False, encoding="utf-8-sig")
        print(f"  脱敏 metrics → {dst / 'metrics_summary.csv'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--dst", required=True, type=Path)
    ap.add_argument("--depth_offset", type=float, default=0.0)
    args = ap.parse_args()
    derive(args.src.resolve(), args.dst.resolve(), depth_offset=args.depth_offset)
    print("\nDONE")


if __name__ == "__main__":
    raise SystemExit(main())
