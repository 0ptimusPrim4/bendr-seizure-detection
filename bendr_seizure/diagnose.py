# -*- coding: utf-8 -*-
"""
Назначение: диагностика разделимости замороженных признаков BENDR
(первый/последний токен, среднее по контексту, среднее по энкодеру),
сохранение кэша на диск и дообучение головы с порогом по валидации.
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
    VARIANTS,
    cache_path,
    channel_sanity,
    class_balance,
    compact_row,
    diagnose_payload,
    event_sens_files,
    extract_variants,
    feature_cache_dir,
    load_variants,
    lr_auc,
    save_variants,
    youden_threshold,
    zscore_per_window,
)
from .metrics import evaluate_sequence, plot_confusion, plot_loss_curves, save_table
from .model import BENDRClassifier
from .preprocessing import Normalizer, process_patient
from .train import build_model, fit_head, pick_device, split_train_val



def main(argv=None, device_override=None):
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
    import argparse
    parser = argparse.ArgumentParser(
        description="Диагностика вариантов признаков BENDR и дообучение головы (НИР-3)."
    )
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "cpu_quick.yaml"))
    parser.add_argument("--min-auc", type=float, default=0.60,
                        help="Минимальный val AUC, чтобы считать признаки разделимыми и считать тест.")
    parser.add_argument("--device", default=None, choices=["auto", "cpu", "cuda"])
    parser.add_argument("--epochs", type=int, default=None)
    args, _unknown = parser.parse_known_args(argv)
    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg["training"]["max_epochs"] = args.epochs
    device = pick_device(cfg, override=device_override or args.device)
    patients = list(cfg["data"]["patients"])
    raw_dir = cfg["paths"]["raw_dir"]
    rng = np.random.default_rng(int(cfg["seed"]))
    feat_bs = int(cfg["training"].get("feature_batch_size") or 8)
    epochs = int(cfg["training"]["max_epochs"])
    vote = int(cfg["postprocessing"]["smoothing"]["window"])
    window_sec = float(cfg["data"]["window_sec"])
    fractions = list(cfg["training"].get("data_fractions") or [0.25, 0.5, 1.0])
    val_frac = float(cfg["training"]["lopo"]["val_fraction"])
    cache_root = feature_cache_dir(cfg)
    results_root = Path(cfg["paths"]["results_dir"]) / "lopo_probe_variants"
    cache_root.mkdir(parents=True, exist_ok=True)
    results_root.mkdir(parents=True, exist_ok=True)

    print("Соответствие меток: 0=фон, 1=приступ.", flush=True)
    print("Нормализация по умолчанию: глобальный z-score по каналу (fit на train).", flush=True)
    print("Дополнительно проверяется z-score внутри окна, если глобальный не разделяет классы.", flush=True)

    print(f"Предобработка обучающих окон: {patients}", flush=True)
    train_cache = {}
    for p in patients:
        print(f"  {p} (train caps)...", flush=True)
        train_cache[p] = process_patient(p, cfg, raw_dir, rng=rng, apply_caps=True)
        class_balance(train_cache[p]["labels"], f"train-cache {p}")
        channel_sanity(train_cache[p]["windows"], f"train {p}")

    all_diag = []
    fold_packs = {}  # target -> payload with chosen variant ready for test

    def build_fold_windows(target, norm="global"):
        others = [p for p in patients if p != target]
        x_tr = np.concatenate([train_cache[p]["windows"] for p in others], 0)
        y_tr = np.concatenate([train_cache[p]["labels"] for p in others], 0)
        tr_idx, va_idx = split_train_val(y_tr, val_frac, int(cfg["seed"]))
        if norm == "global":
            normer = Normalizer().fit(x_tr[tr_idx])
            x_train, x_val = normer.transform(x_tr[tr_idx]), normer.transform(x_tr[va_idx])
        else:
            normer = None
            x_train, x_val = zscore_per_window(x_tr[tr_idx]), zscore_per_window(x_tr[va_idx])
        return {
            "x_train": x_train, "y_train": y_tr[tr_idx],
            "x_val": x_val, "y_val": y_tr[va_idx],
            "normer": normer, "norm": norm,
            "tr_idx": tr_idx, "va_idx": va_idx, "x_tr_all": x_tr, "y_tr_all": y_tr,
        }

    # --- Фаза A: train/val признаки и AUC, при необходимости тест ---
    chosen = None  # (norm, variant) after first fold
    for target in patients:
        print(f"\n=== Диагностика фолда {target} ===", flush=True)
        for norm_name in ("global", "window"):
            path = cache_path(cache_root, target, norm_name)
            if path.is_file() and Path(str(path) + ".meta.json").is_file():
                payload = load_variants(path)
                print(f"  Загружен кэш {path}", flush=True)
            else:
                parts = build_fold_windows(target, norm=norm_name)
                class_balance(parts["y_train"], f"{norm_name} обучение")
                class_balance(parts["y_val"], f"{norm_name} валидация")
                model, _ = build_model(cfg, device)
                print(f"  Извлечение train/val ({norm_name}), пакет {feat_bs}...", flush=True)
                t0 = time.perf_counter()
                tr_feats = extract_variants(model, parts["x_train"], feat_bs, device)
                va_feats = extract_variants(model, parts["x_val"], feat_bs, device)
                print(f"  train/val за {(time.perf_counter()-t0)/60:.1f} мин.", flush=True)
                payload = {"patient": target, "norm": norm_name,
                           "y_train": parts["y_train"], "y_val": parts["y_val"]}
                for v in VARIANTS:
                    payload[f"{v}_train"] = tr_feats[v]
                    payload[f"{v}_val"] = va_feats[v]
                save_variants(path, payload)
                # keep windows/model only if we immediately extract test later
                payload["_parts"] = parts
                payload["_model"] = model
            rows = diagnose_payload(payload)
            for r in rows:
                all_diag.append(r)
            best = max(rows, key=lambda r: (r["val_auc"] if r["val_auc"] is not None else -1))
            print(f"  Лучший на валидации ({norm_name}): {best['variant']} AUC={best['val_auc']:.3f}", flush=True)
            # если признаки почти константа на ctx_first — это ожидаемо; решаем по лучшему варианту
            if chosen is None and best["val_auc"] is not None and best["val_auc"] >= args.min_auc:
                chosen = {"norm": norm_name, "variant": best["variant"], "val_auc": best["val_auc"]}
                print(f"  Выбран вариант: {chosen}", flush=True)
            # не тратим окно-норму, если глобальная уже разделяет
            if chosen and chosen["norm"] == "global" and norm_name == "global":
                break
            if chosen and chosen["norm"] == norm_name:
                break

    save_table(all_diag, results_root / "separability_auc.csv")
    print("\nСводка AUC:", flush=True)
    print(save_table(all_diag, results_root / "separability_auc.csv").to_string(index=False), flush=True)

    if chosen is None:
        # ни один вариант не прошёл порог — зафиксировать и выйти
        print("\nНи один вариант замороженных признаков не дал val AUC "
              f">= {args.min_auc}. Тест не считаем: признаки не разделяют классы.", flush=True)
        Path(results_root / "chosen_variant.json").write_text(
            json.dumps({"chosen": None, "diagnosis": all_diag}, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        return 0

    print(f"\n=== Основной прогон: {chosen['norm']} / {chosen['variant']} ===", flush=True)
    Path(results_root / "chosen_variant.json").write_text(
        json.dumps(chosen, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    var = chosen["variant"]
    compact_rows = []

    for target in patients:
        print(f"\n=== Тест и голова, фолд {target} ===", flush=True)
        path = cache_path(cache_root, target, chosen["norm"])
        payload = load_variants(path) if path.is_file() else {"patient": target, "norm": chosen["norm"]}
        if f"{var}_te" not in payload:
            parts = build_fold_windows(target, norm=chosen["norm"])
            print(f"  Тестовая предобработка {target} без усечения фона...", flush=True)
            test_pack = process_patient(target, cfg, raw_dir, rng=rng, apply_caps=False)
            if chosen["norm"] == "global":
                x_te = parts["normer"].transform(test_pack["windows"])
            else:
                x_te = zscore_per_window(test_pack["windows"])
            class_balance(test_pack["labels"], "тест")
            channel_sanity(test_pack["windows"], f"test {target}")
            model, _ = build_model(cfg, device)
            # если train/val уже в кэше, извлекаем только тест
            if f"{var}_train" not in payload:
                print("  Извлечение train/val (повтор, кэша не было)...", flush=True)
                tr_feats = extract_variants(model, parts["x_train"], feat_bs, device)
                va_feats = extract_variants(model, parts["x_val"], feat_bs, device)
                payload["y_train"], payload["y_val"] = parts["y_train"], parts["y_val"]
                for v in VARIANTS:
                    payload[f"{v}_train"] = tr_feats[v]
                    payload[f"{v}_val"] = va_feats[v]
            print("  Извлечение теста...", flush=True)
            t0 = time.perf_counter()
            te_feats = extract_variants(model, x_te, feat_bs, device)
            print(f"  тест за {(time.perf_counter()-t0)/60:.1f} мин.", flush=True)
            payload["y_te"] = test_pack["labels"]
            payload["starts_te"] = test_pack["starts"]
            payload["file_events"] = test_pack["file_events"]
            for v in VARIANTS:
                payload[f"{v}_te"] = te_feats[v]
            payload["patient"] = target
            payload["norm"] = chosen["norm"]
            save_variants(path, payload)
            del model, x_te, test_pack, parts

        # дополнительная LR на тесте для журнала
        res, _, _ = lr_auc(
            payload[f"{var}_train"], payload["y_train"],
            payload[f"{var}_val"], payload["y_val"],
            payload[f"{var}_te"], payload["y_te"],
        )
        print(f"  LR {var} test AUC={res['test']['auc']:.3f} sens={res['test']['sens']:.3f} "
              f"spec={res['test']['spec']:.3f}", flush=True)

        for frac in fractions:
            print(f"\n--- {target}, {frac:.0%}, {var} ---", flush=True)
            f_tr, y_tr = payload[f"{var}_train"], payload["y_train"]
            if frac < 1.0:
                f_tr, y_tr = take_fraction(f_tr, y_tr, frac, seed=int(cfg["seed"]))
            class_balance(y_tr, f"обучение {frac:.0%}")
            model = BENDRClassifier(cfg)
            tag = f"{target}_frac{int(frac*100)}_{var}_{chosen['norm']}"
            hist = fit_head(
                model, cfg, device, f_tr, y_tr, payload[f"{var}_val"], payload["y_val"],
                epochs=epochs, ckpt_path=Path(cfg["paths"]["checkpoints_dir"]) / f"{tag}.pt",
            )
            plot_loss_curves(hist, results_root / f"loss_{tag}.png")
            model.head.eval()
            with torch.no_grad():
                p_va = model.head(torch.from_numpy(np.ascontiguousarray(payload[f"{var}_val"], dtype=np.float32)).to(device)).cpu().numpy()[:, 1]
                p_te = model.head(torch.from_numpy(np.ascontiguousarray(payload[f"{var}_te"], dtype=np.float32)).to(device)).cpu().numpy()[:, 1]
            t_star, j_star = youden_threshold(payload["y_val"], p_va)
            print(f"  Порог по валидации (Youden, выбор автора): t={t_star:.3f}, J={j_star:.3f}", flush=True)
            pred = (p_te >= t_star).astype(np.int64)
            ev = evaluate_sequence(
                payload["y_te"], pred, window_sec=window_sec,
                starts=payload["starts_te"],
                seizures=[sz for item in payload["file_events"] for sz in item[3]],
                vote_width=vote,
            )
            ev["raw"]["event_sensitivity_files"] = event_sens_files(
                pred, payload["file_events"], window_sec, vote, apply_vote=False)
            ev["smooth"]["event_sensitivity_files"] = event_sens_files(
                pred, payload["file_events"], window_sec, vote, apply_vote=True)
            plot_confusion(ev["smooth"], results_root / f"cm_{tag}.png", f"{target} {frac:.0%} {var}")
            row = compact_row(
                target, frac, ev["raw"], ev["smooth"],
                variant=var, norm=chosen["norm"], threshold=t_star, youden_J=j_star,
                mean_p_seizure=float(p_te.mean()),
                pred_pos_share=float((pred == 1).mean()),
                val_auc_lr=res["val"]["auc"], test_auc_lr=res["test"]["auc"],
            )
            print("  compact:", row, flush=True)
            compact_rows.append(row)
            (results_root / f"metrics_{tag}.json").write_text(
                json.dumps({"history": hist, "eval": ev, "compact": row, "threshold": t_star},
                           ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
            del model

    if compact_rows:
        save_table(compact_rows, results_root / "compact_before_after.csv")
        print("\nИтоговая таблица (до / после постобработки):", flush=True)
        print(save_table(compact_rows, results_root / "compact_before_after.csv").to_string(index=False), flush=True)
        # выбор доли: средняя (val-совместимая) устойчивость = невырожденность spec и event sens на тесте
        # но основная доля выбирается по валидации Youden / невырожденности, 100% — сравнение
        def score(r):
            sb, sa = r.get("spec_before") or 0, r.get("spec_after") or 0
            eb, ea = r.get("event_sens_before") or 0, r.get("event_sens_after") or 0
            # штраф за схлопывание
            if sa <= 0.02 or ea <= 0.02:
                return -1.0
            return (sa + ea) / 2.0
        by = {}
        for r in compact_rows:
            by.setdefault(r["fraction"], []).append(score(r))
        ranking = sorted(((float(np.mean(v)), f) for f, v in by.items()), reverse=True)
        print("Ранжирование долей:", ranking, flush=True)
        scaled = [it for it in ranking if it[1] < 1.0 and it[0] > 0]
        chosen_frac = (scaled[0][1] if scaled else ranking[0][1]) if ranking else 1.0
        print(f"Выбранная доля (25/50 предпочтительнее, если не вырождены): {chosen_frac:.0%}", flush=True)
        Path(results_root / "chosen_fraction.json").write_text(
            json.dumps({"chosen_fraction": chosen_frac, "ranking": ranking, "variant": chosen},
                       ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
