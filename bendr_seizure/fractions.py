# -*- coding: utf-8 -*-
"""
Назначение: после одного извлечения признаков BENDR на фолд LOPO
дообучить голову на 25 %, 50 % и 100 % обучающих окон.
Дата: 23.11.2026
"""
from __future__ import annotations

import json
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch

from .config import PROJECT_ROOT, load_config
from .dataset import take_fraction
from .features import (
    class_balance,
    compact_row,
    event_sens_files,
    apply_norm,
    feature_cache_dir,
    feature_variant,
    norm_scope,
    load_fold_cache,
    save_fold_cache,
    threshold_mode,
    youden_threshold,
)
from .metrics import evaluate_sequence, plot_confusion, plot_loss_curves, save_table
from .preprocessing import Normalizer, process_patient
from .train import build_model, extract_features, fit_head, pick_device, split_train_val


def rec_from_eval(ev, *, patient, fraction, n_train, y_te, mean_p1, stage):
    block = ev[stage]
    return {
        "patient": patient,
        "fraction": fraction,
        "stage": stage,  # raw = до постобработки, smooth = после голосования
        "n_test": int(len(y_te)),
        "n_test_seizure": int((y_te == 1).sum()),
        "n_test_background": int((y_te == 0).sum()),
        "interictal_hours": block.get("interictal_hours"),
        "n_train": int(n_train),
        "sensitivity": block.get("sensitivity"),
        "specificity": block.get("specificity"),
        "fp_per_hour": block.get("fp_per_hour"),
        "grouped_fp_per_hour": block.get("grouped_fp_per_hour"),
        "event_sensitivity": block.get("event_sensitivity_files"),
        "mean_p_seizure": mean_p1,
        "tp": block.get("tp"),
        "tn": block.get("tn"),
        "fp": block.get("fp"),
        "fn": block.get("fn"),
    }


def score_history(hist):
    """Чем выше балл, тем устойчивее сходимость (не вырожденный зонд)."""
    if not hist:
        return -1e9
    last = hist[-1]
    vs = last.get("val_sens")
    vp = last.get("val_spec")
    p1 = last.get("val_p1")
    vl = last.get("val_loss")
    if vs is None or vp is None or vl is None:
        return -1e9
    # штраф за схлопывание к одному классу
    collapse = 0.0
    if vs < 0.05 or vp < 0.05:
        collapse -= 5.0
    if p1 is not None and (p1 < 0.05 or p1 > 0.95):
        collapse -= 3.0
    # оба класса различимы и лосс не взрывается
    return (min(vs, 1.0) + min(vp, 1.0)) - 0.1 * float(vl) + collapse


def extract_fold(cfg, device, target, train_cache, raw_dir, rng, cache_path, feat_bs):
    if cache_path.is_file() and Path(str(cache_path) + ".meta.json").is_file():
        print(f"  Кэш признаков найден: {cache_path}", flush=True)
        return load_fold_cache(cache_path)

    others = [p for p in train_cache if p != target]
    x_tr = np.concatenate([train_cache[p]["windows"] for p in others], 0)
    y_tr = np.concatenate([train_cache[p]["labels"] for p in others], 0)
    val_frac = float(cfg["training"]["lopo"]["val_fraction"])
    tr_idx, va_idx = split_train_val(y_tr, val_frac, int(cfg["seed"]))
    scope = norm_scope(cfg)
    if scope == "window":
        normer = None
    else:
        normer = Normalizer().fit(x_tr[tr_idx])
    x_train, x_val = apply_norm(x_tr[tr_idx], scope, normer), apply_norm(x_tr[va_idx], scope, normer)
    print(f"  Тестовая предобработка {target} без усечения фона (norm={scope})...", flush=True)
    test_pack = process_patient(target, cfg, raw_dir, rng=rng, apply_caps=False)
    x_te = apply_norm(test_pack["windows"], scope, normer)
    y_te = test_pack["labels"]
    class_balance(y_tr[tr_idx], "обучение 100%")
    class_balance(y_tr[va_idx], "валидация")
    class_balance(y_te, "тест (без усечения фона)")

    model, _ = build_model(cfg, device)
    variant = feature_variant(cfg)
    print(f"  Извлечение признаков BENDR, variant={variant} (пакет {feat_bs})...", flush=True)
    t0 = time.perf_counter()
    f_train = extract_features(model, x_train, feat_bs, device, variant=variant)
    f_val = extract_features(model, x_val, feat_bs, device, variant=variant)
    f_te = extract_features(model, x_te, feat_bs, device, variant=variant)
    print(f"  Извлечение заняло {(time.perf_counter()-t0)/60:.1f} мин.", flush=True)
    del model, x_train, x_val, x_te
    payload = {
        "patient": target,
        "f_train": f_train,
        "y_train": y_tr[tr_idx],
        "f_val": f_val,
        "y_val": y_tr[va_idx],
        "f_te": f_te,
        "y_te": y_te,
        "starts": test_pack["starts"],
        "file_events": test_pack["file_events"],
    }
    save_fold_cache(cache_path, payload)
    print(f"  Кэш записан: {cache_path}", flush=True)
    del test_pack
    return payload


def train_one_fraction(cfg, device, pack, frac, results_root, epochs, vote, window_sec):
    from .model import BENDRClassifier

    f_tr, y_tr = pack["f_train"], pack["y_train"]
    if float(frac) < 1.0:
        f_tr, y_tr = take_fraction(f_tr, y_tr, frac, seed=int(cfg["seed"]))
    class_balance(y_tr, f"обучение {frac:.0%}")
    class_balance(pack["y_val"], "валидация")

    # новая голова на тех же замороженных размерах
    model = BENDRClassifier(cfg)
    # веса энкодера не нужны: учим только head на признаках
    tag = f"{pack['patient']}_frac{int(frac*100)}"
    hist = fit_head(
        model, cfg, device, f_tr, y_tr, pack["f_val"], pack["y_val"],
        epochs=epochs, ckpt_path=Path(cfg["paths"]["checkpoints_dir"]) / f"{tag}_cached.pt",
    )
    plot_loss_curves(hist, results_root / f"loss_{tag}.png")
    model.head.eval()
    with torch.no_grad():
        p_va = model.head(torch.from_numpy(np.ascontiguousarray(pack["f_val"], dtype=np.float32)).to(device)).cpu().numpy()
        probs = model.head(torch.from_numpy(np.ascontiguousarray(pack["f_te"], dtype=np.float32)).to(device)).cpu().numpy()
    mean_p1 = float(probs[:, 1].mean())
    mode = threshold_mode(cfg)
    t_star = j_star = None
    if mode == "youden":
        t_star, j_star = youden_threshold(pack["y_val"], p_va[:, 1])
        print(f"  Порог Youden (выбор автора): t={t_star:.3f}, J={j_star:.3f}", flush=True)
        pred = (probs[:, 1] >= t_star).astype(int)
    else:
        pred = probs.argmax(1)
    print(f"  {tag}: mean P(приступ)={mean_p1:.4f}, доля предск. класса 1={float((pred==1).mean()):.3f}", flush=True)

    ev = evaluate_sequence(
        pack["y_te"], pred, window_sec=window_sec,
        starts=pack["starts"],
        seizures=[sz for item in pack["file_events"] for sz in item[3]],
        vote_width=vote,
    )
    ev["raw"]["event_sensitivity_files"] = event_sens_files(
        pred, pack["file_events"], window_sec, vote, apply_vote=False
    )
    ev["smooth"]["event_sensitivity_files"] = event_sens_files(
        pred, pack["file_events"], window_sec, vote, apply_vote=True
    )
    plot_confusion(ev["smooth"], results_root / f"cm_{tag}.png", f"{pack['patient']}, {frac:.0%}")
    row_raw = rec_from_eval(ev, patient=pack["patient"], fraction=frac, n_train=len(y_tr),
                            y_te=pack["y_te"], mean_p1=mean_p1, stage="raw")
    row_sm = rec_from_eval(ev, patient=pack["patient"], fraction=frac, n_train=len(y_tr),
                           y_te=pack["y_te"], mean_p1=mean_p1, stage="smooth")
    compact = compact_row(pack["patient"], frac, ev["raw"], ev["smooth"])
    compact["mean_p_seizure"] = mean_p1
    compact["threshold_mode"] = threshold_mode(cfg)
    compact["threshold"] = t_star
    compact["youden_J"] = j_star
    compact["feature_variant"] = feature_variant(cfg)
    compact["val_score"] = score_history(hist)
    compact["val_sens"] = hist[-1].get("val_sens") if hist else None
    compact["val_spec"] = hist[-1].get("val_spec") if hist else None
    compact["val_loss"] = hist[-1].get("val_loss") if hist else None
    compact["val_p1"] = hist[-1].get("val_p1") if hist else None
    compact["epochs_run"] = hist[-1]["epoch"] if hist else 0
    (results_root / f"metrics_{tag}.json").write_text(
        json.dumps({"history": hist, "eval": ev, "row_raw": row_raw, "row_smooth": row_sm, "compact": compact},
                   ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    del model
    return compact, row_raw, row_sm, hist


def main(argv=None, device_override=None):
    import argparse
    parser = argparse.ArgumentParser(
        description="Дообучение головы на кэше признаков BENDR для долей 25/50/100 %% (НИР-3)."
    )
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "cpu_quick.yaml"))
    parser.add_argument("--fractions", default="0.25,0.5,1.0")
    parser.add_argument("--device", default=None, choices=["auto", "cpu", "cuda"])
    parser.add_argument("--epochs", type=int, default=None)
    args, _unknown = parser.parse_known_args(argv)
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
    cfg = load_config(args.config)
    cfg["training"]["data_fractions"] = [float(x) for x in args.fractions.split(",") if x.strip()]
    if args.epochs is not None:
        cfg["training"]["max_epochs"] = args.epochs
    cfg["paths"]["experiment_dir"] = cfg["paths"].get("experiment_dir") or "lopo_probe"
    # отдельная папка, чтобы не перезаписать текущий 100%-прогон
    results_root = Path(cfg["paths"]["results_dir"]) / str(cfg["paths"].get("fractions_dir") or "lopo_probe_fractions")
    cache_root = feature_cache_dir(cfg)
    results_root.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)

    device = pick_device(cfg, override=device_override or args.device)
    patients = list(cfg["data"]["patients"])
    raw_dir = cfg["paths"]["raw_dir"]
    rng = np.random.default_rng(int(cfg["seed"]))
    feat_bs = int(cfg["training"].get("feature_batch_size") or 8)
    epochs = int(cfg["training"]["max_epochs"])
    vote = int(cfg["postprocessing"]["smoothing"]["window"])
    window_sec = float(cfg["data"]["window_sec"])
    fractions = list(cfg["training"]["data_fractions"])

    print(f"Предобработка обучающих окон (с ограничением фона): {patients}", flush=True)
    print(f"Доли: {fractions}. Кэш: {cache_root}", flush=True)
    train_cache = {}
    for p in patients:
        print(f"  {p} (train caps)...", flush=True)
        train_cache[p] = process_patient(p, cfg, raw_dir, rng=rng, apply_caps=True)
        class_balance(train_cache[p]["labels"], f"train-cache {p}")

    compact_rows, raw_rows, smooth_rows = [], [], []
    extract_times = []
    for target in patients:
        print(f"\n=== Фолд тест: {target} ===", flush=True)
        t0 = time.perf_counter()
        pack = extract_fold(cfg, device, target, train_cache, raw_dir, rng,
                            cache_root / f"{target}_fold.npz", feat_bs)
        dt = time.perf_counter() - t0
        extract_times.append(dt)
        print(f"  Подготовка фолда {target}: {dt/60:.1f} мин.", flush=True)
        if dt > 40 * 60 and not (cache_root / f"{target}_fold.npz").is_file():
            print("  Извлечение дольше 40 мин и кэша нет — останавливаемся.", flush=True)
            break
        for frac in fractions:
            print(f"\n--- {target}, доля {frac:.0%} ---", flush=True)
            compact, row_raw, row_sm, hist = train_one_fraction(
                cfg, device, pack, frac, results_root, epochs, vote, window_sec
            )
            print("  compact:", compact, flush=True)
            compact_rows.append(compact)
            raw_rows.append(row_raw)
            smooth_rows.append(row_sm)

    if compact_rows:
        save_table(compact_rows, results_root / "compact_before_after.csv")
        save_table(raw_rows, results_root / "metrics_before.csv")
        save_table(smooth_rows, results_root / "metrics_after.csv")
        print("\nСводная таблица (до / после постобработки):")
        print(save_table(compact_rows, results_root / "compact_before_after.csv").to_string(index=False))

        # выбор доли по средней устойчивости на валидации
        by_frac = {}
        for r in compact_rows:
            by_frac.setdefault(r["fraction"], []).append(r["val_score"])
        ranking = []
        for frac, scores in by_frac.items():
            ranking.append((float(np.mean(scores)), frac))
        ranking.sort(reverse=True)
        print("\nРанжирование долей по устойчивости валидации (выше лучше):", ranking, flush=True)
        # 100 % только как сравнение: если 25/50 не хуже и не вырождены — берём лучшую из них
        scaled = [item for item in ranking if item[1] < 1.0]
        chosen = ranking[0][1]
        if scaled and ranking[0][1] == 1.0:
            # если 100% лучшая, всё равно предпочтём 25/50, если она не схлопнулась
            best_scaled = scaled[0]
            if best_scaled[0] > 0:
                chosen = best_scaled[1]
        print(f"Выбранная доля для основного результата: {chosen:.0%}", flush=True)
        Path(results_root / "chosen_fraction.json").write_text(
            json.dumps({"chosen": chosen, "ranking": ranking}, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
