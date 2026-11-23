# -*- coding: utf-8 -*-
"""
Назначение: варианты замороженных признаков BENDR, кэш на диск, порог Юдена.
Дата: 23.11.2026

Четыре представления 512-d из одного прямого прохода:
  ctx_first — первый токен контекстуализатора (стартовый, в официальной голове);
  ctx_last  — последний токен (есть в комментарии official features_forward);
  ctx_mean  — среднее по токенам без стартового;
  enc_mean  — среднее по выходу свёрточного энкодера.

Нормализация окон: global (z-score по обучающей выборке, диссертация)
или window (z-score внутри каждого окна, выбор автора).
Порог youden считается на валидации (выбор автора); argmax — порог 0,5.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler

VARIANTS = ("ctx_first", "ctx_last", "ctx_mean", "enc_mean")


def feature_variant(cfg, default="ctx_first"):
    """Ключ feature_variant (корень, features.variant или training.feature_variant)."""
    if cfg.get("feature_variant"):
        return str(cfg["feature_variant"])
    feats = cfg.get("features") or {}
    if feats.get("variant"):
        return str(feats["variant"])
    tr = cfg.get("training") or {}
    if tr.get("feature_variant"):
        return str(tr["feature_variant"])
    return str(default)


def norm_scope(cfg, default="global"):
    """Ключ norm: global (по обучающей выборке) или window (внутри окна)."""
    if cfg.get("norm"):
        return str(cfg["norm"])
    sec = (cfg.get("data") or {}).get("normalization") or {}
    if sec.get("scope"):
        return str(sec["scope"])
    return str(default)


def threshold_mode(cfg, default="argmax"):
    """Режим порога: youden (по валидации) или argmax (класс с большей вероятностью)."""
    post = cfg.get("postprocessing") or {}
    sec = post.get("threshold") or {}
    if sec.get("mode"):
        return str(sec["mode"])
    if post.get("threshold_mode"):
        return str(post["threshold_mode"])
    return str(default)


def feature_cache_dir(cfg):
    feats = cfg.get("features") or {}
    raw = feats.get("cache_dir") or (cfg.get("paths") or {}).get("feature_cache_dir")
    if raw:
        p = Path(raw)
        if not p.is_absolute():
            from .config import PROJECT_ROOT
            p = PROJECT_ROOT / p
        return p
    return Path(cfg["paths"]["results_dir"]) / "feature_cache"


def zscore_per_window(x):
    """Z-score по каждому каналу внутри окна (вариант нормализации, выбор автора)."""
    x = np.asarray(x, dtype=np.float64)
    m = x.mean(axis=-1, keepdims=True)
    s = x.std(axis=-1, keepdims=True)
    s[s < 1e-8] = 1.0
    return np.ascontiguousarray((x - m) / s, dtype=np.float32)


def apply_norm(windows, scope, normer=None):
    """Нормализует окна: scope=global требует обученный Normalizer, scope=window — нет."""
    scope = str(scope).lower()
    if scope == "window":
        return zscore_per_window(windows)
    if scope in ("global", "z-score", "channel"):
        if normer is None:
            raise RuntimeError("Для norm=global нужен Normalizer, обученный на train.")
        return normer.transform(windows)
    raise ValueError(f"Неизвестный способ нормализации: {scope}. Ожидается global или window.")


def class_balance(labels, name):
    y = np.asarray(labels)
    n1 = int((y == 1).sum())
    n0 = int((y == 0).sum())
    n = n0 + n1
    share = (n1 / n) if n else 0.0
    print(f"  {name}: N={n}, приступ={n1}, фон={n0}, доля приступа={share:.3f}", flush=True)
    return n, n1, n0, share


def channel_sanity(windows, name):
    """Проверка, что каналы не занулены и SCALE не константа."""
    x = np.asarray(windows)
    std_ch = x.std(axis=(0, 2))
    zero = [i for i, s in enumerate(std_ch) if s < 1e-8]
    print(
        f"  Каналы {name}: C={x.shape[1]}, std min/med/max="
        f"{std_ch.min():.4g}/{np.median(std_ch):.4g}/{std_ch.max():.4g}, "
        f"нулевые слоты={zero or 'нет'}, SCALE std={std_ch[-1]:.4g}",
        flush=True,
    )
    return std_ch, zero


@torch.no_grad()
def extract_variants(model, windows, bs, device, log_every=40):
    """Один прямой проход: четыре варианта признаков 512-d."""
    model.eval()
    bags = {k: [] for k in VARIANTS}
    x_all = torch.from_numpy(np.ascontiguousarray(windows, dtype=np.float32))
    n = len(x_all)
    t0 = time.perf_counter()
    for i in range(0, n, bs):
        batch = x_all[i:i + bs].to(device)
        encoded = model.encoder(batch)
        bags["enc_mean"].append(encoded.mean(dim=-1).cpu().numpy())
        context = model.contextualizer(encoded)
        bags["ctx_first"].append(context[:, :, 0].cpu().numpy())
        bags["ctx_last"].append(context[:, :, -1].cpu().numpy())
        # среднее без стартового токена (он почти константа, см. журнал попыток)
        if context.shape[-1] > 1:
            bags["ctx_mean"].append(context[:, :, 1:].mean(dim=-1).cpu().numpy())
        else:
            bags["ctx_mean"].append(context.mean(dim=-1).cpu().numpy())
        done = min(i + bs, n)
        step = i // bs + 1
        if log_every and (step % int(log_every) == 0 or done == n):
            dt = time.perf_counter() - t0
            rate = done / dt if dt > 0 else 0.0
            eta = (n - done) / rate if rate > 0 else 0.0
            print(f"    признаки {done}/{n} ({rate:.1f} окн/с, осталось ~{eta/60:.1f} мин)", flush=True)
    return {k: np.concatenate(v, 0) for k, v in bags.items()}


def feat_stats(x, y, name):
    std = x.std(axis=0)
    norms = np.linalg.norm(x, axis=1)
    mu0 = x[y == 0].mean(0) if (y == 0).any() else np.zeros(x.shape[1])
    mu1 = x[y == 1].mean(0) if (y == 1).any() else np.zeros(x.shape[1])
    print(
        f"  {name}: std_по_выборке min/med/max={std.min():.3g}/{np.median(std):.3g}/{std.max():.3g}, "
        f"L2 std={norms.std():.3g}, ||mu1-mu0||={np.linalg.norm(mu1 - mu0):.4g}",
        flush=True,
    )
    return {
        "std_min": float(std.min()),
        "std_med": float(np.median(std)),
        "std_max": float(std.max()),
        "centroid_dist": float(np.linalg.norm(mu1 - mu0)),
    }


def lr_auc(x_tr, y_tr, x_va, y_va, x_te=None, y_te=None, seed=42):
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(x_tr)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=int(seed))
    clf.fit(Xtr, y_tr)
    out = {}
    for split, X, y in (("train", Xtr, y_tr), ("val", scaler.transform(x_va), y_va)):
        proba = clf.predict_proba(X)[:, 1]
        pred = (proba >= 0.5).astype(int)
        auc = float(roc_auc_score(y, proba)) if len(np.unique(y)) > 1 else None
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        sen = tp / (tp + fn) if (tp + fn) else None
        spe = tn / (tn + fp) if (tn + fp) else None
        out[split] = {"auc": auc, "sens": sen, "spec": spe, "mean_p": float(proba.mean())}
    if x_te is not None:
        proba = clf.predict_proba(scaler.transform(x_te))[:, 1]
        auc = float(roc_auc_score(y_te, proba)) if len(np.unique(y_te)) > 1 else None
        pred = (proba >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_te, pred, labels=[0, 1]).ravel()
        sen = tp / (tp + fn) if (tp + fn) else None
        spe = tn / (tn + fp) if (tn + fp) else None
        out["test"] = {"auc": auc, "sens": sen, "spec": spe, "mean_p": float(proba.mean())}
    return out, clf, scaler


def youden_threshold(y, proba, grid=101):
    """Порог по валидации: максимум чувствительность+специфичность−1 (выбор автора)."""
    y = np.asarray(y)
    proba = np.asarray(proba)
    best_t, best_j = 0.5, -2.0
    for t in np.linspace(0.0, 1.0, grid):
        pred = (proba >= t).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        tn = int(((pred == 0) & (y == 0)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        sen = tp / (tp + fn) if (tp + fn) else 0.0
        spe = tn / (tn + fp) if (tn + fp) else 0.0
        j = sen + spe - 1.0
        if j > best_j:
            best_t, best_j = float(t), float(j)
    return best_t, best_j


def probs_to_pred(probs, mode="argmax", threshold=None):
    """Метки окон по вероятностям: argmax или порог Youden / заданный threshold."""
    probs = np.asarray(probs)
    mode = str(mode).lower()
    if mode == "youden" or threshold is not None and mode != "argmax":
        if threshold is None:
            raise ValueError("Для режима youden нужно передать порог, найденный на валидации.")
        return (probs[:, 1] >= float(threshold)).astype(np.int64)
    return probs.argmax(axis=1).astype(np.int64)


def event_sens_files(pred, file_events, window_sec, vote_width, apply_vote):
    from .postprocessing import event_sensitivity, majority_voting
    offset = 0
    hits, total = 0, 0
    for name, starts, labels, seizures in file_events:
        n = len(labels)
        chunk = pred[offset:offset + n]
        offset += n
        if not seizures:
            continue
        sm = majority_voting(chunk, vote_width) if apply_vote else chunk
        evs = event_sensitivity(sm, starts, window_sec, seizures)
        if evs is None:
            continue
        total += len(seizures)
        hits += evs * len(seizures)
    if total == 0:
        return None
    return hits / total


def compact_row(patient, fraction, ev_raw, ev_smooth, **extra):
    rec = {
        "patient": patient,
        "fraction": fraction,
        "event_sens_before": ev_raw.get("event_sensitivity_files"),
        "event_sens_after": ev_smooth.get("event_sensitivity_files"),
        "spec_before": ev_raw.get("specificity"),
        "spec_after": ev_smooth.get("specificity"),
        "fp_h_before": ev_raw.get("fp_per_hour"),
        "fp_h_after": ev_smooth.get("fp_per_hour"),
        "grouped_fp_h_before": ev_raw.get("grouped_fp_per_hour"),
        "grouped_fp_h_after": ev_smooth.get("grouped_fp_per_hour"),
        "interictal_hours": ev_smooth.get("interictal_hours"),
    }
    rec.update(extra)
    return rec


def cache_path(root, target, norm_name):
    return Path(root) / f"{target}_{norm_name}_variants.npz"


def save_variants(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrs = {}
    for split in ("train", "val", "te"):
        if f"y_{split}" in payload:
            arrs[f"y_{split}"] = payload[f"y_{split}"]
        for v in VARIANTS:
            key = f"{v}_{split}"
            if key in payload:
                arrs[key] = payload[key]
        if f"starts_{split}" in payload:
            arrs[f"starts_{split}"] = payload[f"starts_{split}"]
    np.savez_compressed(path, **arrs)
    meta = {
        "patient": payload["patient"],
        "norm": payload.get("norm"),
        "splits": [s for s in ("train", "val", "te") if f"y_{s}" in payload],
        "variants": list(VARIANTS),
    }
    if "file_events" in payload:
        meta["file_events"] = [
            (name, np.asarray(starts).tolist(), np.asarray(labels).tolist(), events)
            for name, starts, labels, events in payload["file_events"]
        ]
    Path(str(path) + ".meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    print(f"  Кэш записан: {path} ({path.stat().st_size / 1e6:.1f} МБ, split={meta['splits']})", flush=True)


def load_variants(path):
    path = Path(path)
    data = np.load(path)
    meta = json.loads(Path(str(path) + ".meta.json").read_text(encoding="utf-8"))
    payload = {"patient": meta["patient"], "norm": meta.get("norm")}
    for k in data.files:
        payload[k] = data[k]
    if "file_events" in meta:
        payload["file_events"] = [
            (name, np.asarray(starts), np.asarray(labels), events)
            for name, starts, labels, events in meta["file_events"]
        ]
    return payload


def diagnose_payload(payload, splits=("train", "val"), seed=42):
    rows = []
    have_te = "y_te" in payload and "ctx_mean_te" in payload
    for var in VARIANTS:
        x_tr, y_tr = payload[f"{var}_train"], payload["y_train"]
        x_va, y_va = payload[f"{var}_val"], payload["y_val"]
        feat_stats(x_tr, y_tr, f"{var}/train")
        kw = {}
        if have_te:
            kw = {"x_te": payload[f"{var}_te"], "y_te": payload["y_te"]}
        res, _, _ = lr_auc(x_tr, y_tr, x_va, y_va, seed=seed, **kw)
        rec = {"variant": var, "norm": payload.get("norm"), "patient": payload["patient"]}
        for sp, d in res.items():
            rec[f"{sp}_auc"] = d["auc"]
            rec[f"{sp}_sens"] = d["sens"]
            rec[f"{sp}_spec"] = d["spec"]
        extra = f"  test AUC={res['test']['auc']:.3f}" if "test" in res else ""
        print(
            f"  LR {var}: train AUC={res['train']['auc']:.3f}  val AUC={res['val']['auc']:.3f}"
            + extra
            + f"  val sens={res['val']['sens']:.3f} spec={res['val']['spec']:.3f}",
            flush=True,
        )
        rows.append(rec)
    return rows


def save_fold_cache(path, payload):
    """Кэш одного фолда в формате run_fractions_from_cache (один вариант признаков)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        f_train=payload["f_train"],
        y_train=payload["y_train"],
        f_val=payload["f_val"],
        y_val=payload["y_val"],
        f_te=payload["f_te"],
        y_te=payload["y_te"],
        starts=payload["starts"],
    )
    meta = {
        "patient": payload["patient"],
        "file_events": [
            (name, np.asarray(starts).tolist(), np.asarray(labels).tolist(), events)
            for name, starts, labels, events in payload["file_events"]
        ],
    }
    Path(str(path) + ".meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")


def load_fold_cache(path):
    path = Path(path)
    data = np.load(path)
    meta = json.loads(Path(str(path) + ".meta.json").read_text(encoding="utf-8"))
    file_events = []
    for name, starts, labels, events in meta["file_events"]:
        file_events.append((name, np.asarray(starts), np.asarray(labels), events))
    return {
        "patient": meta["patient"],
        "f_train": data["f_train"],
        "y_train": data["y_train"],
        "f_val": data["f_val"],
        "y_val": data["y_val"],
        "f_te": data["f_te"],
        "y_te": data["y_te"],
        "starts": data["starts"],
        "file_events": file_events,
    }
