# -*- coding: utf-8 -*-
"""
Назначение: точка входа — дымовой тест, синтетическая проверка, LOPO и оценка чекпоинта.
Дата: 23.11.2026
"""

from __future__ import annotations
import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

from .config import PROJECT_ROOT, load_config
from .dataset import make_loader, take_fraction
from .features import (
    apply_norm,
    event_sens_files,
    feature_variant,
    norm_scope,
    probs_to_pred,
    threshold_mode,
    youden_threshold,
)
from .metrics import evaluate_sequence, plot_confusion, plot_loss_curves, save_table
from .preprocessing import Normalizer, load_edf, locate_edf, locate_summary, parse_summary, process_patient
from .train import (
    Trainer,
    batch_size,
    build_model,
    extract_features,
    fit_head,
    pick_device,
    split_train_val,
)


def _apply_device(cfg, device_override=None):
    if device_override:
        cfg["device"] = device_override
    return pick_device(cfg, override=device_override)


def cmd_smoke(cfg, device_override=None):
    """Загрузка одного EDF и прямой проход модели с официальными весами."""
    raw_dir = Path(cfg["paths"]["raw_dir"])
    patient = cfg["data"]["patients"][0]
    summary = locate_summary(raw_dir, patient)
    if summary is None:
        raise FileNotFoundError(
            f"Нет сводки {patient} в {raw_dir}. Сначала скачайте данные (см. README.md)."
        )
    _, records = parse_summary(summary)
    rec = next((r for r in records if r.seizures), records[0])
    path = locate_edf(raw_dir, patient, rec.file_name)
    if path is None:
        raise FileNotFoundError("Нет EDF для дымового теста.")
    data, names, sfreq = load_edf(path, float(cfg["data"]["sampling_rate_hz"]))
    print(f"Файл: {path}")
    print(f"Каналы ({len(names)}): {list(names)}")
    print(f"Частота дискретизации: {sfreq} Гц, форма сигнала: {tuple(data.shape)}")

    device = _apply_device(cfg, device_override)
    model, report = build_model(cfg, device)
    print("Загрузка весов:")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    n_par = sum(p.numel() for p in model.parameters())
    print(f"Параметров в модели: {n_par}")

    dummy = torch.randn(
        2,
        int(cfg["data"]["n_channels_out"]),
        int(cfg["data"]["window_sec"] * cfg["data"]["sampling_rate_hz"]),
        device=device,
    )
    model.eval()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(dummy)
    dt = time.perf_counter() - t0
    print(
        f"Прямой проход dummy [2, 20, 1024]: выход {tuple(out.shape)}, "
        f"сумма по классам {out.sum(dim=-1).tolist()}"
    )
    print(f"Время прямого прохода: {dt:.3f} с, устройство: {device}")
    return 0


def _synthetic_batch(n, cfg, seed=0):
    """Синтетические окна: фон — узкополосный шум, приступ — импульсы большей амплитуды."""
    rng = np.random.default_rng(seed)
    c = int(cfg["data"]["n_channels_out"])
    t = int(cfg["data"]["window_sec"] * cfg["data"]["sampling_rate_hz"])
    y = rng.integers(0, 2, size=n).astype(np.int64)
    x = rng.normal(0, 1.0, size=(n, c, t)).astype(np.float32)
    for i, lab in enumerate(y):
        if lab == 1:
            # Короткие «спайки» в середине окна.
            mid = t // 2
            x[i, :, mid - 40:mid + 40] += 4.0 * rng.normal(0, 1, size=(c, 80)).astype(np.float32)
    return x, y


def cmd_mock(cfg, epochs=2, device_override=None):
    """Синтетический прогон: проверка, что лосс считается и градиент идёт."""
    device = _apply_device(cfg, device_override)
    model, _ = build_model(cfg, device)
    x, y = _synthetic_batch(64, cfg, seed=int(cfg["seed"]))
    tr, va = split_train_val(y, 0.25, int(cfg["seed"]))
    norm = Normalizer().fit(x[tr])
    xtr, xva = norm.transform(x[tr]), norm.transform(x[va])
    bs = batch_size(cfg, device)
    trainer = Trainer(model, cfg, device)
    results = Path(cfg["paths"]["results_dir"]) / "mock"
    hist = trainer.fit(
        make_loader(xtr, y[tr], bs, weighted=True, seed=cfg["seed"]),
        make_loader(xva, y[va], bs, shuffle=False, seed=cfg["seed"]),
        epochs=epochs,
        ckpt_path=results / "mock.pt",
    )
    plot_loss_curves(hist, results / "loss.png")
    probs = trainer.predict(xva, bs)
    pred = probs.argmax(1)
    ev = evaluate_sequence(y[va], pred, window_sec=cfg["data"]["window_sec"])
    print("Синтетика, валидация после голосования:", ev["smooth"])
    plot_confusion(ev["smooth"], results / "confusion.png", "Синтетическая выборка")
    Path(results / "metrics.json").write_text(
        json.dumps(ev, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return 0


def _predict_events(pred, file_events, window_sec, vote_width):
    """Собирает событийную чувствительность по файлам пациента."""
    from .postprocessing import event_sensitivity, majority_voting
    offset = 0
    hits, total = 0, 0
    for name, starts, labels, seizures in file_events:
        n = len(labels)
        chunk = pred[offset:offset + n]
        offset += n
        if not seizures:
            continue
        sm = majority_voting(chunk, vote_width)
        evs = event_sensitivity(sm, starts, window_sec, seizures)
        if evs is None:
            continue
        total += len(seizures)
        hits += evs * len(seizures)
    if total == 0:
        return None
    return hits / total


def _class_balance(labels, name):
    y = np.asarray(labels)
    n1 = int((y == 1).sum())
    n0 = int((y == 0).sum())
    n = n0 + n1
    share = (n1 / n) if n else 0.0
    print(f"  {name}: N={n}, приступ={n1}, фон={n0}, доля приступа={share:.3f}", flush=True)
    return n, n1, n0, share


def _normalizer_payload(norm):
    return {
        "normalizer_mean": np.asarray(norm.mean_).tolist(),
        "normalizer_std": np.asarray(norm.std_).tolist(),
    }


def _load_normalizer(state, fallback_fit=None):
    """Восстанавливает Normalizer из чекпоинта или, если статистик нет, из fallback_fit()."""
    norm = Normalizer()
    if state and state.get("normalizer_mean") is not None and state.get("normalizer_std") is not None:
        norm.mean_ = np.asarray(state["normalizer_mean"], dtype=np.float64)
        norm.std_ = np.asarray(state["normalizer_std"], dtype=np.float64)
        return norm
    if fallback_fit is None:
        raise RuntimeError(
            "В чекпоинте нет статистик нормализатора, а пересчитать их не из чего. "
            "Запустите оценку при доступных обучающих записях тех же пациентов."
        )
    return fallback_fit()


def cmd_lopo(cfg, device_override=None):
    """Leave-one-patient-out на указанном в конфиге подмножестве пациентов."""
    device = _apply_device(cfg, device_override)
    patients = list(cfg["data"]["patients"])
    raw_dir = cfg["paths"]["raw_dir"]
    rng = np.random.default_rng(int(cfg["seed"]))
    mode = str(cfg["training"].get("mode") or "full")
    print(f"Предобработка обучающих окон (с ограничением фона): {patients}", flush=True)
    print(f"Режим обучения: {mode}", flush=True)
    print(f"Устройство: {device}", flush=True)
    print("Соответствие меток: класс 0 — фон, класс 1 — приступ (SSWCE смотрит на P(класс=1)).", flush=True)
    cache = {}
    for p in patients:
        print(f"  {p} (train caps)...", flush=True)
        cache[p] = process_patient(p, cfg, raw_dir, rng=rng, apply_caps=True)
        d = cache[p]
        _class_balance(d["labels"], f"train-cache {p}")

    results_root = Path(cfg["paths"]["results_dir"]) / str(cfg["paths"].get("experiment_dir") or "lopo")
    results_root.mkdir(parents=True, exist_ok=True)
    fractions = list(cfg["training"]["data_fractions"])
    rows = []
    histories = {}
    bs = batch_size(cfg, device)
    vote = int(cfg["postprocessing"]["smoothing"]["window"])
    window_sec = float(cfg["data"]["window_sec"])
    epochs = int(cfg["training"]["max_epochs"])
    val_frac = float(cfg["training"]["lopo"]["val_fraction"])
    ckpt_dir = Path(cfg["paths"]["checkpoints_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for frac in fractions:
        print(f"\n=== Доля обучающих данных: {frac:.0%} ===", flush=True)
        for target in patients:
            print(f"Тест: {target}", flush=True)
            others = [p for p in patients if p != target]
            x_tr = np.concatenate([cache[p]["windows"] for p in others], 0)
            y_tr = np.concatenate([cache[p]["labels"] for p in others], 0)
            x_tr, y_tr = take_fraction(x_tr, y_tr, frac, seed=int(cfg["seed"]))
            tr_idx, va_idx = split_train_val(y_tr, val_frac, int(cfg["seed"]))
            scope = norm_scope(cfg)
            if scope == "window":
                norm = None
            else:
                norm = Normalizer().fit(x_tr[tr_idx])
            x_train, x_val = apply_norm(x_tr[tr_idx], scope, norm), apply_norm(x_tr[va_idx], scope, norm)
            print(f"  Тестовая предобработка {target} без усечения фона (norm={scope})...", flush=True)
            test_pack = process_patient(target, cfg, raw_dir, rng=rng, apply_caps=False)
            x_te = apply_norm(test_pack["windows"], scope, norm)
            y_te = test_pack["labels"]
            _class_balance(y_tr[tr_idx], "обучение")
            _class_balance(y_tr[va_idx], "валидация")
            _class_balance(y_te, "тест (без усечения фона)")

            model, _ = build_model(cfg, device)
            tag = f"{target}_frac{int(frac * 100)}"
            feat_bs = int(cfg["training"].get("feature_batch_size") or 8)
            extra = {
                **(_normalizer_payload(norm) if norm is not None else {}),
                "patient": target,
                "fraction": frac,
                "mode": mode,
                "feature_variant": feature_variant(cfg),
                "norm": scope,
                "threshold_mode": threshold_mode(cfg),
            }
            sen_lr = spe_lr = None
            trainer = None
            if mode == "head_only":
                variant = feature_variant(cfg)
                print(f"  Извлечение признаков BENDR, variant={variant} (пакет {feat_bs})...", flush=True)
                f_train = extract_features(model, x_train, feat_bs, device, variant=variant)
                f_val = extract_features(model, x_val, feat_bs, device, variant=variant)
                f_te = extract_features(model, x_te, feat_bs, device, variant=variant)
                # Диагностика разделимости замороженных признаков линейным классификатором
                # (не входит в официальные метрики, только в журнал).
                try:
                    from sklearn.linear_model import LogisticRegression
                    from sklearn.metrics import confusion_matrix
                    clf = LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=int(cfg["seed"]),
                    )
                    clf.fit(f_train, y_tr[tr_idx])
                    pred_lr = clf.predict(f_te)
                    tn, fp, fn, tp = confusion_matrix(y_te, pred_lr, labels=[0, 1]).ravel()
                    sen_lr = tp / (tp + fn) if (tp + fn) else None
                    spe_lr = tn / (tn + fp) if (tn + fp) else None
                    print(
                        f"  Диагностика LogisticRegression: sens={sen_lr}, spec={spe_lr}, "
                        f"tp={tp} tn={tn} fp={fp} fn={fn}",
                        flush=True,
                    )
                except Exception as exc:
                    print(f"  Диагностика LR не удалась: {exc}", flush=True)
                hist = fit_head(
                    model, cfg, device, f_train, y_tr[tr_idx], f_val, y_tr[va_idx],
                    epochs=epochs, ckpt_path=ckpt_dir / f"{tag}.pt", extra=extra,
                )
                model.head.eval()
                with torch.no_grad():
                    probs = model.head(
                        torch.from_numpy(np.ascontiguousarray(f_te, dtype=np.float32)).to(device)
                    )
                    probs = probs.cpu().numpy()
                # f_val нужен для порога Youden; удалим после оценки.
            else:
                trainer = Trainer(model, cfg, device)
                trainer.ckpt_extra = extra
                hist = trainer.fit(
                    make_loader(x_train, y_tr[tr_idx], bs, weighted=True, seed=cfg["seed"]),
                    make_loader(x_val, y_tr[va_idx], bs, shuffle=False, seed=cfg["seed"]),
                    epochs=epochs,
                    ckpt_path=ckpt_dir / f"{tag}.pt",
                )
                probs = trainer.predict(x_te, bs)
            plot_loss_curves(hist, results_root / f"loss_{tag}.png")
            histories[tag] = hist
            thr_mode = threshold_mode(cfg)
            t_star = j_star = None
            if thr_mode == "youden":
                # Для головы считаем вероятность на валидации; для full — на val-окнах.
                if mode == "head_only":
                    model.head.eval()
                    with torch.no_grad():
                        p_va = model.head(
                            torch.from_numpy(np.ascontiguousarray(f_val, dtype=np.float32)).to(device)
                        ).cpu().numpy()[:, 1]
                    y_va = y_tr[va_idx]
                else:
                    p_va = trainer.predict(x_val, bs)[:, 1]
                    y_va = y_tr[va_idx]
                t_star, j_star = youden_threshold(y_va, p_va)
                print(f"  Порог Youden (выбор автора): t={t_star:.3f}, J={j_star:.3f}", flush=True)
                pred = probs_to_pred(probs, mode="youden", threshold=t_star)
            else:
                pred = probs_to_pred(probs, mode="argmax")
            mean_p1 = float(probs[:, 1].mean())
            print(
                f"  Средняя P(приступ) на тесте: {mean_p1:.4f}; "
                f"доля предсказаний класса 1: {float((pred == 1).mean()):.3f}",
                flush=True,
            )
            ev = evaluate_sequence(
                y_te, pred, window_sec=window_sec,
                starts=test_pack["starts"],
                seizures=[sz for item in test_pack["file_events"] for sz in item[3]],
                vote_width=vote,
            )
            # Событийная метрика корректнее считается по файлам (свои шкалы времени).
            ev["raw"]["event_sensitivity_files"] = event_sens_files(
                pred, test_pack["file_events"], window_sec, vote, apply_vote=False
            )
            ev["smooth"]["event_sensitivity_files"] = event_sens_files(
                pred, test_pack["file_events"], window_sec, vote, apply_vote=True
            )
            if t_star is not None:
                ckpt = ckpt_dir / f"{tag}.pt"
                if ckpt.is_file():
                    state = torch.load(ckpt, map_location="cpu", weights_only=False)
                    state["threshold"] = t_star
                    state["youden_J"] = j_star
                    torch.save(state, ckpt)
            plot_confusion(ev["smooth"], results_root / f"cm_{tag}.png",
                           f"{target}, {frac:.0%} данных")
            rec = {
                "patient": target,
                "fraction": frac,
                "n_test": int(len(y_te)),
                "n_test_seizure": int((y_te == 1).sum()),
                "n_test_background": int((y_te == 0).sum()),
                "interictal_hours": ev["smooth"].get("interictal_hours"),
                "n_train": int(len(tr_idx)),
                "sensitivity": ev["smooth"]["sensitivity"],
                "specificity": ev["smooth"]["specificity"],
                "fp_per_hour": ev["smooth"]["fp_per_hour"],
                "grouped_fp_per_hour": ev["smooth"].get("grouped_fp_per_hour"),
                "event_sensitivity": ev["smooth"].get("event_sensitivity_files"),
                "mean_p_seizure": mean_p1,
                "feature_variant": feature_variant(cfg),
                "norm": scope,
                "threshold_mode": thr_mode,
                "threshold": t_star,
                "youden_J": j_star,
                "lr_sensitivity": sen_lr,
                "lr_specificity": spe_lr,
                "tp": ev["smooth"]["tp"], "tn": ev["smooth"]["tn"],
                "fp": ev["smooth"]["fp"], "fn": ev["smooth"]["fn"],
            }
            print("  ", rec, flush=True)
            rows.append(rec)
            (results_root / f"metrics_{tag}.json").write_text(
                json.dumps({"history": hist, "eval": ev, "row": rec}, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
            # Освобождаем граф перед следующим фолдом.
            del model
            if trainer is not None:
                del trainer
            del x_te, test_pack
            if mode == "head_only":
                del f_train, f_val
            if device.type == "cuda":
                torch.cuda.empty_cache()

    df = save_table(rows, results_root / "lopo_results.csv")
    print("\nСводная таблица:")
    print(df.to_string(index=False))
    summary = []
    for frac in fractions:
        part = [r for r in rows if r["fraction"] == frac]

        def avg(key, _part=part):
            vals = [r[key] for r in _part if r[key] is not None]
            return float(np.mean(vals)) if vals else None

        summary.append({
            "fraction": frac,
            "sensitivity": avg("sensitivity"),
            "specificity": avg("specificity"),
            "fp_per_hour": avg("fp_per_hour"),
            "grouped_fp_per_hour": avg("grouped_fp_per_hour"),
            "event_sensitivity": avg("event_sensitivity"),
        })
    print("\nСредние по долям:")
    print(save_table(summary, results_root / "lopo_summary.csv").to_string(index=False))
    return 0


_CKPT_RE = re.compile(r"(chb\d+)_frac(\d+)", re.I)


def _parse_ckpt_tag(path):
    m = _CKPT_RE.search(Path(path).stem)
    if not m:
        return None, None
    return m.group(1).lower(), int(m.group(2)) / 100.0


def cmd_eval(cfg, checkpoint=None, device_override=None, patient=None):
    """Оценка уже обученного чекпоинта с постобработкой, без дообучения.

    Если --checkpoint не задан, разбираются все *.pt из paths.checkpoints_dir.
    Нормализатор берётся из чекпоинта (новые прогоны его сохраняют). Если
    статистик нет (старые CPU-чекпоинты), они пересчитываются по остальным
    пациентам из конфига с тем же seed.
    """
    device = _apply_device(cfg, device_override)
    ckpt_dir = Path(cfg["paths"]["checkpoints_dir"])
    if checkpoint:
        ckpts = [Path(checkpoint)]
    else:
        ckpts = sorted(ckpt_dir.glob("*.pt"))
    if not ckpts:
        raise FileNotFoundError(
            f"Не найдены чекпоинты. Укажите --checkpoint или положите *.pt в {ckpt_dir}."
        )
    for p in ckpts:
        if not p.is_file():
            raise FileNotFoundError(f"Нет файла чекпоинта: {p}")

    results_root = Path(cfg["paths"]["results_dir"]) / str(
        cfg["paths"].get("experiment_dir") or "lopo_eval"
    )
    results_root.mkdir(parents=True, exist_ok=True)
    vote = int(cfg["postprocessing"]["smoothing"]["window"])
    window_sec = float(cfg["data"]["window_sec"])
    mode = str(cfg["training"].get("mode") or "full")
    feat_bs = int(cfg["training"].get("feature_batch_size") or 8)
    bs = batch_size(cfg, device)
    raw_dir = cfg["paths"]["raw_dir"]
    rows = []

    # Кэш предобработки пациентов накапливается по мере надобности.
    cache = {}
    rng = np.random.default_rng(int(cfg["seed"]))

    def get_pack(pid, apply_caps):
        key = (pid, bool(apply_caps))
        if key not in cache:
            cache[key] = process_patient(pid, cfg, raw_dir, rng=rng, apply_caps=apply_caps)
        return cache[key]

    for ckpt_path in ckpts:
        print(f"\nОценка чекпоинта: {ckpt_path}", flush=True)
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        ckpt_patient, ckpt_frac = _parse_ckpt_tag(ckpt_path)
        target = (patient or state.get("patient") or ckpt_patient)
        frac = state.get("fraction", ckpt_frac)
        if target is None:
            raise ValueError(
                f"Не удалось определить тестового пациента для {ckpt_path.name}. "
                "Задайте --patient или используйте имя вида chb01_frac100.pt."
            )
        ckpt_mode = state.get("mode") or mode
        others = [p for p in cfg["data"]["patients"] if p != target]

        def fit_norm():
            if not others:
                raise RuntimeError(
                    "Нельзя пересчитать нормализатор: в конфиге нет других пациентов."
                )
            print("  Статистик нормализатора в чекпоинте нет — пересчитываю по обучающим пациентам.", flush=True)
            parts_x, parts_y = [], []
            for pid in others:
                pack = get_pack(pid, True)
                parts_x.append(pack["windows"])
                parts_y.append(pack["labels"])
            x_tr = np.concatenate(parts_x, 0)
            y_tr = np.concatenate(parts_y, 0)
            if frac is not None:
                x_tr, y_tr = take_fraction(x_tr, y_tr, float(frac), seed=int(cfg["seed"]))
            tr_idx, _ = split_train_val(y_tr, float(cfg["training"]["lopo"]["val_fraction"]), int(cfg["seed"]))
            return Normalizer().fit(x_tr[tr_idx])

        scope = state.get("norm") or norm_scope(cfg)
        if scope == "window":
            norm = None
        else:
            norm = _load_normalizer(state, fallback_fit=fit_norm)
        test_pack = get_pack(target, False)
        x_te = apply_norm(test_pack["windows"], scope, norm)
        y_te = test_pack["labels"]
        _class_balance(y_te, f"тест {target}")

        model, _ = build_model(cfg, device)
        if "head" in state and "model" not in state:
            model.head.load_state_dict(state["head"])
        elif "model" in state:
            missing, unexpected = model.load_state_dict(state["model"], strict=False)
            if missing or unexpected:
                print(f"  load_state_dict: missing={list(missing)}, unexpected={list(unexpected)}", flush=True)
        elif "head" in state:
            model.head.load_state_dict(state["head"])
        else:
            raise KeyError(f"В чекпоинте {ckpt_path.name} нет ключей head/model.")
        model.eval()

        variant = state.get("feature_variant") or feature_variant(cfg)
        if ckpt_mode == "head_only":
            feats = extract_features(model, x_te, feat_bs, device, variant=variant)
            with torch.no_grad():
                probs = model.head(
                    torch.from_numpy(np.ascontiguousarray(feats, dtype=np.float32)).to(device)
                ).cpu().numpy()
        else:
            trainer = Trainer(model, cfg, device)
            probs = trainer.predict(x_te, bs)
            del trainer

        thr_mode = state.get("threshold_mode") or threshold_mode(cfg)
        t_star = state.get("threshold")
        if thr_mode == "youden" and t_star is None:
            print("  В чекпоинте нет порога Youden — использую argmax (0,5).", flush=True)
            thr_mode = "argmax"
        pred = probs_to_pred(probs, mode=thr_mode, threshold=t_star)
        ev = evaluate_sequence(
            y_te, pred, window_sec=window_sec,
            starts=test_pack["starts"],
            seizures=[sz for item in test_pack["file_events"] for sz in item[3]],
            vote_width=vote,
        )
        ev["smooth"]["event_sensitivity_files"] = _predict_events(
            pred, test_pack["file_events"], window_sec, vote
        )
        # Метрики «до» уже лежат в ev['raw'].
        tag = f"{target}_frac{int((frac or 1.0) * 100)}"
        plot_confusion(ev["smooth"], results_root / f"cm_{tag}.png",
                       f"{target}, оценка чекпоинта")
        rec = {
            "patient": target,
            "fraction": frac,
            "checkpoint": str(ckpt_path),
            "n_test": int(len(y_te)),
            "n_test_seizure": int((y_te == 1).sum()),
            "n_test_background": int((y_te == 0).sum()),
            "interictal_hours": ev["smooth"].get("interictal_hours"),
            "sensitivity": ev["smooth"]["sensitivity"],
            "specificity": ev["smooth"]["specificity"],
            "fp_per_hour": ev["smooth"]["fp_per_hour"],
            "grouped_fp_per_hour": ev["smooth"].get("grouped_fp_per_hour"),
            "event_sensitivity": ev["smooth"].get("event_sensitivity_files"),
            "sensitivity_raw": ev["raw"]["sensitivity"],
            "specificity_raw": ev["raw"]["specificity"],
            "fp_per_hour_raw": ev["raw"]["fp_per_hour"],
            "grouped_fp_per_hour_raw": ev["raw"].get("grouped_fp_per_hour"),
            "tp": ev["smooth"]["tp"], "tn": ev["smooth"]["tn"],
            "fp": ev["smooth"]["fp"], "fn": ev["smooth"]["fn"],
        }
        print("  после голосования:", rec, flush=True)
        print("  до голосования: sens={sensitivity}, spec={specificity}, FP/ч={fp_per_hour}".format(**ev["raw"]), flush=True)
        rows.append(rec)
        (results_root / f"metrics_{tag}.json").write_text(
            json.dumps({"eval": ev, "row": rec}, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        del model, x_te
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if rows:
        print("\nСводная таблица оценки:")
        print(save_table(rows, results_root / "eval_results.csv").to_string(index=False))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Дообучение и оценка BENDR на CHB-MIT."
    )
    parser.add_argument(
        "command",
        choices=["smoke", "mock", "train", "lopo", "eval", "diagnose", "fractions"],
        help="smoke — загрузка EDF и прямой проход; mock — синтетика; "
             "train/lopo — схема LOPO; eval — только оценка чекпоинта; "
             "diagnose — варианты признаков и голова с порогом Youden; "
             "fractions — доли 25/50/100 %% с кэша признаков.",
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "cpu_quick.yaml"),
        help="Путь к YAML-конфигу (по умолчанию configs/cpu_quick.yaml).",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Переопределить training.max_epochs.")
    parser.add_argument(
        "--device",
        default=None,
        choices=["auto", "cpu", "cuda"],
        help="Переопределить устройство (по умолчанию — из конфига, обычно auto).",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Для eval: путь к одному .pt. Если не задан, берутся все из checkpoints_dir.",
    )
    parser.add_argument(
        "--patient",
        default=None,
        help="Для eval: тестовый пациент, если его нельзя вывести из имени чекпоинта.",
    )
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg["training"]["max_epochs"] = args.epochs
    if args.command == "smoke":
        return cmd_smoke(cfg, device_override=args.device)
    if args.command == "mock":
        return cmd_mock(cfg, epochs=args.epochs or 2, device_override=args.device)
    if args.command == "eval":
        return cmd_eval(cfg, checkpoint=args.checkpoint, device_override=args.device, patient=args.patient)
    if args.command == "diagnose":
        from .diagnose import main as diagnose_main
        return diagnose_main(argv, device_override=args.device)
    if args.command == "fractions":
        from .fractions import main as fractions_main
        return fractions_main(argv, device_override=args.device)
    return cmd_lopo(cfg, device_override=args.device)


if __name__ == "__main__":
    sys.exit(main())
