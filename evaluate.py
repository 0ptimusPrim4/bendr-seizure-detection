# -*- coding: utf-8 -*-
"""
Шаг 3. Оценка и пост-обработка по сохранённым предсказаниям train.py.

Для каждого фолда и доли данных считаются метрики ДО пост-обработки (окно
голосования 1) и ПОСЛЕ (голосование большинством в нечётном окне 3/5/7,
группировка подряд идущих ложных срабатываний):
  событийная чувствительность, чувствительность и специфичность по окнам,
  FP/ч по формуле диссертации, FP/ч после группировки, AUC.
Порог — найденный на валидации в train.py (Юден) или заданный флагом --threshold.

Результаты в <run-dir>/:
  metrics_all.csv   все фолды × доли × окна голосования
  summary.csv       среднее и медиана по фолдам для каждой доли и окна
  metrics.json      то же в JSON

Примеры:
  python evaluate.py --run-dir results/lopo_full
  python evaluate.py --run-dir results/lopo_full --vote-widths 1 3 5 7
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import common as C


def evaluate_fold(fold_dir, widths, window_sec, thr_override=None):
    info = json.loads((fold_dir / "fold.json").read_text(encoding="utf-8"))
    z = np.load(fold_dir / "preds.npz")
    p, y = z["p_test"], z["y_test"].astype(np.int64)
    events = {int(k): [tuple(e) for e in v] for k, v in info["events"].items()}
    thr = float(thr_override if thr_override is not None else info["threshold"])
    auc = C.safe_auc(y, p)
    rows = []
    for w in widths:
        m = C.detection_metrics(y, p, z["file_idx"], z["starts"], events, thr, w, window_sec)
        rows.append({"fold": info["fold"], "fraction": info["fraction"],
                     "stage": "до пост-обработки" if w == 1 else "после пост-обработки",
                     "auc": auc, "val_auc": info.get("val_auc"), "freeze": info.get("freeze"),
                     "n_test": int(len(y)), "n_test_seizure": int(y.sum()), **m})
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description="Оценка и пост-обработка (шаг 3 из 3).")
    ap.add_argument("--run-dir", required=True, help="каталог прогона из train.py, например results/lopo_full")
    ap.add_argument("--vote-widths", nargs="+", type=int, default=None,
                    help="нечётные окна голосования; 1 = без пост-обработки (по умолчанию из конфига)")
    ap.add_argument("--threshold", type=float, default=None, help="фиксированный порог вместо порога с валидации")
    args = ap.parse_args(argv)

    run_dir = C.resolve_path(args.run_dir)
    cfg = yaml.safe_load((run_dir / "config_used.yaml").read_text(encoding="utf-8"))
    window_sec = float(cfg["data"]["window_sec"])
    widths = args.vote_widths or [1] + list(cfg["postprocessing"]["vote_widths"])
    widths = sorted(set(int(w) for w in widths))
    if any(w % 2 == 0 for w in widths):
        raise SystemExit("Окна голосования должны быть нечётными.")

    rows = []
    for fold_dir in sorted(d for d in run_dir.iterdir() if (d / "preds.npz").is_file()):
        rows += evaluate_fold(fold_dir, widths, window_sec, args.threshold)
    if not rows:
        print(f"В {run_dir} нет preds.npz — сначала запустите train.py.")
        return 1
    df = pd.DataFrame(rows)
    df.to_csv(run_dir / "metrics_all.csv", index=False)
    cols = ["auc", "event_sensitivity", "sensitivity", "specificity", "fp_per_hour", "grouped_fp_per_hour"]
    summ = (df.groupby(["fraction", "vote_width", "stage"])[cols]
              .agg(["mean", "median"]).round(4))
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ = summ.reset_index()
    summ.insert(3, "n_folds", df.groupby(["fraction", "vote_width"]).size().values)
    summ.to_csv(run_dir / "summary.csv", index=False)
    (run_dir / "metrics.json").write_text(
        json.dumps({"per_fold": rows, "summary": summ.to_dict(orient="records")},
                   ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(summ[["fraction", "vote_width", "n_folds", "auc_mean", "event_sensitivity_mean",
                    "specificity_mean", "fp_per_hour_mean", "grouped_fp_per_hour_mean"]].to_string(index=False))
    print(f"\nТаблицы: {run_dir / 'summary.csv'}, {run_dir / 'metrics_all.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
