# -*- coding: utf-8 -*-
"""
Шаг 2. Дообучение BENDR на кэше окон CHB-MIT (LOPO или patient-wise).

Возможности:
  --freeze full|encoder|backbone|partial  разморозка всего / заморозка энкодера /
                                          только голова / последние N слоёв трансформера
  --fractions 0.25 0.5 1.0                доли обучающей выборки (варианты 25/50/100 %)
  --scheme lopo|patient_wise              схема разбиения
  --balancing sampler|class_weights|none  WeightedRandomSampler или веса классов в SSWCE
  AMP на GPU, лучший чекпоинт по val_loss, ранняя остановка, порог Юдена на валидации.
  --evaluate                              сразу посчитать метрики (вызывает evaluate.py)

Для режима backbone признаки BENDR считаются один раз на фолд, после чего голова
учится на них (быстро даже на CPU).

Примеры:
  python train.py --config configs/gpu.yaml --evaluate
  python train.py --config configs/gpu.yaml --freeze backbone --fractions 0.25 0.5 1.0 --evaluate
  python train.py --config configs/gpu.yaml --test-patients chb01 chb02 --freeze partial --unfreeze-last 2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

import common as C

# weight_norm в контекстуализаторе нужен ради совместимости с весами BENDR
warnings.filterwarnings("ignore", message=".*weight_norm.*")


# ---------------------------- разбиения -------------------------------

def stratified_split(labels, frac, rng):
    """Индексы доли frac с сохранением соотношения классов (остальное — второй массив)."""
    a, b = [], []
    for cls in (0, 1):
        idx = np.where(labels == cls)[0]
        rng.shuffle(idx)
        k = int(round(len(idx) * frac))
        a.append(idx[:k])
        b.append(idx[k:])
    return np.sort(np.concatenate(a)), np.sort(np.concatenate(b))


def items_for(cache_no, cache, file_ids=None, max_bg_per_file=None, rng=None):
    """Пары (номер кэша, индекс окна); фон по желанию урезается в каждом файле (только train)."""
    idx = np.arange(len(cache))
    if file_ids is not None:
        idx = idx[np.isin(cache.file_idx, list(file_ids))]
    if max_bg_per_file:
        keep = []
        for f in np.unique(cache.file_idx[idx]):
            fi = idx[cache.file_idx[idx] == f]
            pos, neg = fi[cache.labels[fi] == 1], fi[cache.labels[fi] == 0]
            if len(neg) > max_bg_per_file:
                neg = np.sort(rng.choice(neg, max_bg_per_file, replace=False))
            keep += [pos, neg]
        idx = np.sort(np.concatenate(keep))
    return np.stack([np.full(len(idx), cache_no), idx], 1)


def make_folds(cfg, caches, scheme, test_patients, rng):
    """Возвращает список фолдов: имя, train/val/test пары и файлы для статистик нормализации."""
    tr_cfg = cfg["training"]
    cap = cfg["data"].get("train_max_background_per_file")
    val_frac = float(tr_cfg["val_fraction"])
    names = list(caches)
    folds = []
    if scheme == "lopo":
        for test in test_patients:
            train_p = [p for p in names if p != test]
            if not train_p:
                raise ValueError("Для LOPO нужно минимум два пациента.")
            trval = np.concatenate([items_for(names.index(p), caches[p], None, cap, rng) for p in train_p])
            y = np.array([caches[names[c]].labels[i] for c, i in trval])
            va, tr = stratified_split(y, val_frac, rng)
            folds.append({
                "name": test, "train": trval[tr], "val": trval[va],
                "test": items_for(names.index(test), caches[test]),
                "norm_files": [(caches[p], None) for p in train_p],
            })
    elif scheme == "patient_wise":
        # Отдельная модель на пациента: тестовые файлы отделяются целиком (выбрано автором:
        # доля test_fraction от файлов с приступами и без, минимум по одному файлу с приступом).
        tf = float(cfg["training"]["patient_wise"]["test_fraction"])
        for p in test_patients:
            c = caches[p]
            sz = [i for i, f in enumerate(c.files) if f["n_seizure"] > 0]
            bg = [i for i, f in enumerate(c.files) if f["n_seizure"] == 0]
            if len(sz) < 2:
                print(f"{p}: меньше двух файлов с приступами, patient-wise невозможен — пропуск")
                continue
            rng.shuffle(sz)
            rng.shuffle(bg)
            k_sz = min(len(sz) - 1, max(1, int(round(len(sz) * tf))))
            k_bg = int(round(len(bg) * tf))
            test_files = set(sz[:k_sz] + bg[:k_bg])
            train_files = [i for i in range(len(c.files)) if i not in test_files]
            no = names.index(p)
            trval = items_for(no, c, train_files, cap, rng)
            va, tr = stratified_split(c.labels[trval[:, 1]], val_frac, rng)
            folds.append({"name": p, "train": trval[tr], "val": trval[va],
                          "test": items_for(no, c, sorted(test_files)),
                          "norm_files": [(c, train_files)]})
    else:
        raise ValueError(f"Неизвестная схема: {scheme}")
    return folds


# ---------------------------- обучение --------------------------------

def make_loader(ds, labels, bs, cfg, train, workers, device, seed):
    g = torch.Generator().manual_seed(seed)
    kw = dict(batch_size=bs, num_workers=workers, pin_memory=device.type == "cuda",
              persistent_workers=workers > 0)
    if train and cfg["training"]["balancing"] == "sampler":
        counts = np.maximum(np.bincount(labels, minlength=2), 1).astype(np.float64)
        n = cfg["training"].get("samples_per_epoch") or len(labels)
        sampler = WeightedRandomSampler(torch.as_tensor(1.0 / counts[labels]), int(n),
                                        replacement=True, generator=g)
        return DataLoader(ds, sampler=sampler, **kw)
    return DataLoader(ds, shuffle=train, generator=g, **kw)


def run_epoch(net, loader, crit, device, opt=None, scaler=None, amp=False, model=None):
    train = opt is not None
    C.train_mode(model or net, train)
    total, seen = 0.0, 0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).long()
            with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                probs = net(x)
            loss = crit(probs, y)       # SSWCE считается в float32
            if train:
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            total += float(loss.detach()) * len(y)
            seen += len(y)
    return total / max(seen, 1)


@torch.no_grad()
def predict(net, loader, device, amp, model=None):
    C.train_mode(model or net, False)
    out = []
    for x, _ in loader:
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            out.append(net(x.to(device, non_blocking=True)).float()[:, 1].cpu().numpy())
    return np.concatenate(out) if out else np.empty(0)


@torch.no_grad()
def extract(model, loader, device, amp):
    """Признаки BENDR (512) для режима backbone; энкодер и контекстуализатор в eval."""
    model.eval()
    feats, ys = [], []
    for x, y in loader:
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            feats.append(model.features(x.to(device, non_blocking=True)).float().cpu())
        ys.append(torch.as_tensor(y))
    return torch.cat(feats), torch.cat(ys)


def fit(net, model, tr_loader, va_loader, cfg, device, amp, ckpt, log):
    t = cfg["training"]
    params = [p for p in net.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=float(t["lr"]), weight_decay=float(t["weight_decay"]))
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=float(t["scheduler_factor"]), patience=int(t["scheduler_patience"]))
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    crit = cfg["_criterion"].to(device)
    best, bad, history = float("inf"), 0, []
    for ep in range(1, int(t["max_epochs"]) + 1):
        t0 = time.time()
        tr = run_epoch(net, tr_loader, crit, device, opt, scaler, amp, model)
        va = run_epoch(net, va_loader, crit, device, amp=amp, model=model)
        sched.step(va)
        history.append({"epoch": ep, "train_loss": tr, "val_loss": va, "lr": opt.param_groups[0]["lr"]})
        mark = ""
        if va < best - float(t["early_stopping_min_delta"]):
            best, bad, mark = va, 0, " *"
            # сохраняются только обучаемые параметры: замороженные совпадают с весами BENDR
            state = {k: v.detach().cpu() for k, v in model.named_parameters() if v.requires_grad}
            torch.save({"trainable": state, "epoch": ep, "val_loss": va}, ckpt)
        else:
            bad += 1
        log(f"  эпоха {ep:3d}: train {tr:.4f}  val {va:.4f}  {time.time() - t0:.0f} с{mark}")
        if bad >= int(t["early_stopping_patience"]):
            log("  ранняя остановка")
            break
    state = torch.load(ckpt, map_location="cpu", weights_only=True)["trainable"]
    model.load_state_dict(state, strict=False)
    return history


def train_fold(fold, frac, cfg, caches, device, args, run_dir, log):
    t = cfg["training"]
    tag = f"{fold['name']}_frac{int(round(frac * 100))}"
    out = run_dir / tag
    if (out / "preds.npz").is_file() and not args.force:
        log(f"{tag}: уже посчитан, пропуск (--force для пересчёта)")
        return
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(cfg["seed"]))
    names = list(caches)
    lab = lambda items: np.array([caches[names[c]].labels[i] for c, i in items], np.int64)
    tr_items, sel = fold["train"], None
    if frac < 1.0:
        sel, _ = stratified_split(lab(tr_items), frac, rng)
        tr_items = tr_items[sel]
    y_tr, y_va, y_te = lab(tr_items), lab(fold["val"]), lab(fold["test"])
    log(f"{tag}: train {len(y_tr)} (приступ {y_tr.sum()}), val {len(y_va)} ({y_va.sum()}), "
        f"test {len(y_te)} ({y_te.sum()})")
    if y_tr.sum() == 0 or y_va.sum() == 0:
        log(f"{tag}: нет окон приступа в train/val — пропуск")
        return

    caches_list = [caches[n] for n in names]
    norm = C.Normalizer.from_caches(cfg["norm"], fold["norm_files"])
    ds = lambda items: C.WindowSet(caches_list, items, norm)
    counts = np.maximum(np.bincount(y_tr, minlength=2), 1)
    cw = (len(y_tr) / (2.0 * counts)).tolist() if t["balancing"] == "class_weights" else None
    cfg["_criterion"] = C.SSWCE(cfg["loss"]["alpha"], cfg["loss"]["beta"], cw)

    model = C.BENDRClassifier(cfg)
    rep = model.load_pretrained(cfg["paths"]["weights_dir"])
    bad_keys = {k: len(v["missing"]) + len(v["unexpected"]) for k, v in rep.items()}
    if any(bad_keys.values()):
        log(f"{tag}: ВНИМАНИЕ, веса BENDR загружены не полностью: {rep}")
    n_tr, n_all = C.configure_freezing(model, t["freeze"], t.get("unfreeze_last", 0))
    model.to(device)
    log(f"{tag}: режим {t['freeze']}, обучаемых параметров {n_tr:,} из {n_all:,}")
    amp = bool(t["amp"]) and device.type == "cuda"
    bs = int(t["batch_size"])
    w = args.workers
    seed = int(cfg["seed"])
    ckpt = out / "best.pt"

    if t["freeze"] == "backbone":
        # признаки один раз, затем учится только голова Linear(512, 2) + softmax
        # признаки фолда считаются один раз и переиспользуются для всех долей данных
        if fold.get("_feats") is None:
            fl = lambda items: DataLoader(ds(items), batch_size=bs, num_workers=w,
                                          pin_memory=device.type == "cuda")
            t0 = time.time()
            fold["_feats"] = [extract(model, fl(fold[k]), device, amp)[0] for k in ("train", "val", "test")]
            log(f"{fold['name']}: признаки BENDR извлечены за {time.time() - t0:.0f} с")
        f_tr, f_va, f_te = fold["_feats"]
        if frac < 1.0:
            f_tr = f_tr[torch.as_tensor(sel)]
        head = model.head
        tl = make_loader(TensorDataset(f_tr, torch.as_tensor(y_tr)), y_tr, bs, cfg, True, 0, device, seed)
        vl = DataLoader(TensorDataset(f_va, torch.as_tensor(y_va)), batch_size=bs)
        tel = DataLoader(TensorDataset(f_te, torch.as_tensor(y_te)), batch_size=bs)
        history = fit(head, model, tl, vl, cfg, device, False, ckpt, log)
        p_va, p_te = predict(head, vl, device, False, model), predict(head, tel, device, False, model)
    else:
        tl = make_loader(ds(tr_items), y_tr, bs, cfg, True, w, device, seed)
        vl = make_loader(ds(fold["val"]), y_va, bs, cfg, False, w, device, seed)
        tel = make_loader(ds(fold["test"]), y_te, bs, cfg, False, w, device, seed)
        history = fit(model, model, tl, vl, cfg, device, amp, ckpt, log)
        p_va, p_te = predict(model, vl, device, amp), predict(model, tel, device, amp)

    # порог: Юден на валидации (выбрано автором) или 0,5 (argmax двух классов)
    if cfg["postprocessing"]["threshold"] == "youden":
        thr, j = C.youden_threshold(y_va, p_va)
    else:
        thr, j = 0.5, None
    # для оценки: номер файла, начало окна и интервалы приступов тестового пациента
    tc = caches[names[int(fold["test"][0, 0])]]
    ti = fold["test"][:, 1]
    events = {int(i): tc.files[i]["seizures"] for i in np.unique(tc.file_idx[ti])}
    np.savez_compressed(out / "preds.npz", p_test=p_te, y_test=y_te, file_idx=tc.file_idx[ti],
                        starts=tc.starts[ti], p_val=p_va, y_val=y_va)
    info = {"fold": fold["name"], "fraction": frac, "scheme": args.scheme, "freeze": t["freeze"],
            "unfreeze_last": t.get("unfreeze_last"), "balancing": t["balancing"], "norm": cfg["norm"],
            "feature": cfg["model"]["feature"], "threshold": thr, "youden_j_val": j,
            "val_auc": C.safe_auc(y_va, p_va), "n_train": int(len(y_tr)),
            "n_train_seizure": int(y_tr.sum()), "events": events,
            "test_files": [tc.files[i]["file"] for i in events], "history": history,
            "normalizer": norm.state()}
    (out / "fold.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"{tag}: порог {thr:.3f}, val AUC {info['val_auc']}, test AUC {C.safe_auc(y_te, p_te)}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Дообучение BENDR (шаг 2 из 3). Флаги переопределяют YAML.")
    ap.add_argument("--config", default="configs/gpu.yaml")
    ap.add_argument("--scheme", choices=["lopo", "patient_wise"], help="схема оценки")
    ap.add_argument("--patients", nargs="+", help="все пациенты эксперимента (train + test)")
    ap.add_argument("--test-patients", nargs="+", help="какие фолды считать (по умолчанию все)")
    ap.add_argument("--fractions", nargs="+", type=float, help="доли обучающей выборки, например 0.25 0.5 1.0")
    ap.add_argument("--freeze", choices=["full", "encoder", "backbone", "partial"])
    ap.add_argument("--unfreeze-last", type=int, help="для partial: сколько последних слоёв трансформера учить")
    ap.add_argument("--balancing", choices=["sampler", "class_weights", "none"])
    ap.add_argument("--norm", choices=["global", "window"])
    ap.add_argument("--feature", choices=["ctx_first", "ctx_mean", "enc_mean"])
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--no-amp", action="store_true", help="отключить смешанную точность")
    ap.add_argument("--device", default=None, help="auto | cuda | cpu")
    ap.add_argument("--workers", type=int, default=None, help="процессов DataLoader")
    ap.add_argument("--threads", type=int, default=None, help="torch.set_num_threads (для CPU)")
    ap.add_argument("--run-name", help="имя каталога в results/ (по умолчанию из конфига)")
    ap.add_argument("--evaluate", action="store_true", help="после обучения запустить evaluate.py")
    ap.add_argument("--force", action="store_true", help="пересчитать уже готовые фолды")
    ap.add_argument("--set", nargs="*", default=[], metavar="КЛЮЧ=ЗНАЧЕНИЕ",
                    help="любой ключ YAML, например training.scheduler_patience=5")
    args = ap.parse_args(argv)

    cfg = C.apply_overrides(C.load_config(args.config), args.set)
    t = cfg["training"]
    # флаги командной строки имеют приоритет над YAML
    for key, val in (("scheme", args.scheme), ("freeze", args.freeze), ("unfreeze_last", args.unfreeze_last),
                     ("balancing", args.balancing), ("max_epochs", args.epochs),
                     ("batch_size", args.batch_size), ("lr", args.lr)):
        if val is not None:
            t[key] = val
    if args.fractions:
        t["data_fractions"] = args.fractions
    if args.norm:
        cfg["norm"] = args.norm
    if args.feature:
        cfg["model"]["feature"] = args.feature
    if args.no_amp:
        t["amp"] = False
    if args.patients:
        cfg["data"]["patients"] = args.patients
    args.scheme = t["scheme"]
    args.workers = int(args.workers if args.workers is not None else t.get("num_workers", 0))
    if args.threads:
        torch.set_num_threads(args.threads)

    C.set_seed(int(cfg["seed"]))
    device = C.pick_device(args.device or cfg.get("device", "auto"))
    cache_dir = C.resolve_path(cfg["paths"]["cache_dir"])
    caches = {p: C.PatientCache(cache_dir, p) for p in cfg["data"]["patients"]}
    run_name = args.run_name or cfg["paths"].get("run_name") or f"{t['scheme']}_{t['freeze']}"
    run_dir = C.resolve_path(cfg["paths"]["results_dir"]) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    logf = (run_dir / "train.log").open("a", encoding="utf-8")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    saved = {k: v for k, v in cfg.items() if not k.startswith("_")}
    (run_dir / "config_used.yaml").write_text(C.yaml.safe_dump(saved, allow_unicode=True, sort_keys=False),
                                              encoding="utf-8")
    log(f"=== {time.strftime('%Y-%m-%d %H:%M')} | устройство {device} | "
        f"{torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'} | run {run_dir}")
    tests = args.test_patients or cfg["data"]["patients"]
    folds = make_folds(cfg, caches, t["scheme"], tests, np.random.default_rng(int(cfg["seed"])))
    for fold in folds:
        for frac in t["data_fractions"]:
            train_fold(fold, float(frac), cfg, caches, device, args, run_dir, log)
            if device.type == "cuda":
                torch.cuda.empty_cache()
    logf.close()
    if args.evaluate:
        import evaluate
        evaluate.main(["--run-dir", str(run_dir)])
    return 0


if __name__ == "__main__":
    sys.exit(main())
