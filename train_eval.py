# -*- coding: utf-8 -*-
"""
Дообучение BENDR на CHB-MIT и оценка с пост-обработкой — одной командой:

  python train_eval.py                 # подготовит данные при необходимости, LOPO по chb01–chb23

Если кэша нет, вызывается prepare_data.py (скачивание CHB-MIT и весов, предобработка).
Режимы --freeze: full — учится всё (диссертация); encoder — заморожен свёрточный энкодер;
partial — учатся последние --unfreeze-last слоёв трансформера и голова; backbone — только голова.
Доли 25/50/100 % (--fractions) — объём обучающей выборки; веса BENDR есть только для полной сети.
Метрики до и после пост-обработки (голосование в нечётном окне, группировка FP) —
results/<run>/summary.csv, metrics_all.csv, metrics.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import prepare_data as P

P.require("torch", "sklearn", "pandas")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader, Dataset, TensorDataset, WeightedRandomSampler  # noqa: E402

warnings.filterwarnings("ignore", message=".*weight_norm.*")


# ------------------------------------------------- модель BENDR (структура SPOClab-ca)

class ChannelDropout(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        return x * torch.bernoulli(x.new_full((x.shape[0], x.shape[1], 1), 1 - self.p))


class Identity(nn.Module):
    def forward(self, x):
        return x


class Permute(nn.Module):
    def __init__(self, *dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims)


class Encoder(nn.Module):
    # ядра (3,2,2,2,2,2) в официальном коде делаются нечётными → (3,3,3,3,3,3), шаги (3,2,2,2,2,2)
    def __init__(self, c_in=20, h=512, widths=(3, 2, 2, 2, 2, 2), strides=(3, 2, 2, 2, 2, 2), p=0.5):
        super().__init__()
        self.encoder = nn.Sequential()
        for i, (w, s) in enumerate(zip(widths, strides)):
            w = w if w % 2 else w + 1
            self.encoder.add_module(f"Encoder_{i}", nn.Sequential(
                nn.Conv1d(c_in if i == 0 else h, h, w, stride=s, padding=w // 2),
                ChannelDropout(p), nn.GroupNorm(h // 2, h), nn.GELU()))

    def forward(self, x):
        return self.encoder(x)


class Contextualizer(nn.Module):
    def __init__(self, h=512, d=1536, ff=3076, layers=8, heads=8, pos_k=25, pos_g=16, dropout=0.15):
        super().__init__()
        conv = nn.utils.weight_norm(nn.Conv1d(h, h, pos_k, padding=pos_k // 2, groups=pos_g), dim=2)
        self.relative_position = nn.Sequential(conv, nn.GELU())
        self.input_conditioning = nn.Sequential(Permute(0, 2, 1), nn.LayerNorm(h), nn.Dropout(dropout),
                                                Permute(0, 2, 1), nn.Conv1d(h, d, 1), Permute(2, 0, 1))
        self.transformer_layers = nn.ModuleList()
        for _ in range(layers):
            layer = nn.TransformerEncoderLayer(d, heads, ff, dropout, activation="gelu")
            layer.norm1, layer.norm2 = Identity(), Identity()   # T-Fixup, как в BENDR
            self.transformer_layers.append(layer)
        self.norm = nn.LayerNorm(d)
        self.mask_replacement = nn.Parameter(torch.zeros(h))
        self.output_layer = nn.Conv1d(d, h, 1)

    def forward(self, x):
        x = self.input_conditioning(x + self.relative_position(x))
        x = torch.cat([x.new_full((1, x.shape[1], x.shape[2]), -5.0), x], 0)  # стартовый токен
        for layer in self.transformer_layers:
            x = layer(x)
        return self.output_layer(x.permute(1, 2, 0))


class BENDR(nn.Module):
    def __init__(self, feature="ctx_first"):
        super().__init__()
        self.feature = feature
        self.encoder, self.contextualizer = Encoder(), Contextualizer()
        self.head = nn.Sequential(nn.Linear(512, 2), nn.Softmax(-1))

    def features(self, x):
        z = self.encoder(x)
        if self.feature == "enc_mean":
            return z.mean(-1)
        c = self.contextualizer(z)
        return c[:, :, 0] if self.feature == "ctx_first" else c[:, :, 1:].mean(-1)

    def forward(self, x):
        return self.head(self.features(x))

    def load_pretrained(self, wdir):
        for name, mod in (("encoder", self.encoder), ("contextualizer", self.contextualizer)):
            miss, unexp = mod.load_state_dict(torch.load(Path(wdir) / f"{name}.pt", map_location="cpu",
                                                         weights_only=True), strict=False)
            if miss or unexp:
                print(f"ВНИМАНИЕ: {name}.pt загружен не полностью: {miss} {unexp}")


def set_freeze(model, mode, k):
    for p in model.parameters():
        p.requires_grad = True
    ctx = model.contextualizer
    frozen = {"full": [], "encoder": [model.encoder], "backbone": [model.encoder, ctx],
              "partial": [model.encoder, ctx.relative_position, ctx.input_conditioning,
                          *ctx.transformer_layers[:len(ctx.transformer_layers) - k]]}[mode]
    if mode == "partial":
        ctx.mask_replacement.requires_grad = False
    for m in frozen:
        for p in m.parameters():
            p.requires_grad = False
    model._frozen = frozen
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_mode(model, flag):
    model.train(flag)
    for m in model._frozen:   # замороженные части без dropout
        m.eval()


class SSWCE(nn.Module):
    """L = CE + α(1 − спец.) + β(1 − чувств.), α = 0,3, β = 0,5 (Benfenati, ур. 5.9)."""

    def __init__(self, alpha=0.3, beta=0.5, weights=None):
        super().__init__()
        self.a, self.b = alpha, beta
        self.w = None if weights is None else torch.tensor(weights, dtype=torch.float32)

    def forward(self, probs, y):
        probs = probs.float()
        w = None if self.w is None else self.w.to(probs.device)
        ce = F.nll_loss(torch.log(probs.clamp(min=1e-7)), y, weight=w)
        p1, yf = probs[:, 1], y.float()
        sens = (p1 * yf).sum() / yf.sum().clamp(min=1)
        spec = ((1 - p1) * (1 - yf)).sum() / (1 - yf).sum().clamp(min=1)
        return ce + self.a * (1 - spec) + self.b * (1 - sens)


# ------------------------------------------------------------------ данные

class Windows(Dataset):
    def __init__(self, caches, items, norm):
        self.caches, self.items, self.norm = caches, items, norm
        self.labels = np.array([caches[c].labels[i] for c, i in items], np.int64)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, k):
        c, i = self.items[k]
        x = self.caches[c].window(i)
        if self.norm is None:   # z-score внутри окна
            s = x.std(-1, keepdims=True)
            x = (x - x.mean(-1, keepdims=True)) / np.where(s < 1e-8, 1, s)
        else:                   # глобальный z-score по обучающим пациентам
            x = (x - self.norm[0]) / self.norm[1]
        return torch.from_numpy(x.astype(np.float32)), int(self.labels[k])


def global_norm(parts):
    s = q = 0.0
    n = 0
    for cache, files in parts:
        a, b, c = cache.channel_stats(files)
        s, q, n = s + a, q + b, n + c
    m = s / n
    return (m[:, None].astype(np.float32), np.sqrt(np.maximum(q / n - m ** 2, 1e-20))[:, None].astype(np.float32))


def strat(labels, frac, rng):
    a, b = [], []
    for c in (0, 1):
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        k = int(round(len(idx) * frac))
        a.append(idx[:k])
        b.append(idx[k:])
    return np.sort(np.concatenate(a)), np.sort(np.concatenate(b))


def items(no, cache, files=None, cap=None, rng=None):
    idx = np.arange(len(cache))
    if files is not None:
        idx = idx[np.isin(cache.file_idx, list(files))]
    if cap:  # урезать фон в обучающих файлах
        keep = []
        for f in np.unique(cache.file_idx[idx]):
            fi = idx[cache.file_idx[idx] == f]
            neg = fi[cache.labels[fi] == 0]
            if len(neg) > cap:
                neg = rng.choice(neg, cap, replace=False)
            keep += [fi[cache.labels[fi] == 1], neg]
        idx = np.sort(np.concatenate(keep))
    return np.stack([np.full(len(idx), no), idx], 1)


def make_folds(a, caches, rng):
    names = list(caches)
    folds = []
    for test in a.test_patients or names:
        no = names.index(test)
        if a.scheme == "lopo":
            train_p = [p for p in names if p != test]
            tv = np.concatenate([items(names.index(p), caches[p], None, a.max_bg, rng) for p in train_p])
            test_items, norm_parts = items(no, caches[test]), [(caches[p], None) for p in train_p]
        else:  # patient_wise: доля файлов пациента (≥1 с приступом) уходит в тест — выбрано автором
            c = caches[test]
            sz = [i for i, f in enumerate(c.files) if f["n_seizure"]]
            bg = [i for i, f in enumerate(c.files) if not f["n_seizure"]]
            if len(sz) < 2:
                print(f"{test}: меньше двух файлов с приступами — пропуск")
                continue
            rng.shuffle(sz)
            rng.shuffle(bg)
            tf = set(sz[:min(len(sz) - 1, max(1, round(len(sz) * a.test_fraction)))] + bg[:round(len(bg) * a.test_fraction)])
            trf = [i for i in range(len(c.files)) if i not in tf]
            tv, test_items, norm_parts = items(no, c, trf, a.max_bg, rng), items(no, c, sorted(tf)), [(c, trf)]
        y = np.array([caches[names[c]].labels[i] for c, i in tv])
        va, tr = strat(y, a.val_fraction, rng)
        folds.append({"name": test, "train": tv[tr], "val": tv[va], "test": test_items, "norm": norm_parts})
    return folds


# --------------------------------------------------------------- обучение

def loader(ds, y, a, train, device, workers):
    kw = dict(batch_size=a.batch_size, num_workers=workers, pin_memory=device.type == "cuda")
    if train and a.balancing == "sampler":   # WeightedRandomSampler, веса обратны частоте класса
        cnt = np.maximum(np.bincount(y, minlength=2), 1)
        s = WeightedRandomSampler(torch.as_tensor(1.0 / cnt[y]), a.samples_per_epoch or len(y), replacement=True)
        return DataLoader(ds, sampler=s, **kw)
    return DataLoader(ds, shuffle=train, **kw)


def run(net, model, dl, crit, device, amp, opt=None, scaler=None):
    train_mode(model, opt is not None)
    tot = n = 0
    with torch.set_grad_enabled(opt is not None):
        for x, y in dl:
            x, y = x.to(device, non_blocking=True), y.to(device).long()
            with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                loss = crit(net(x), y)
            if opt is not None:
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            tot, n = tot + float(loss) * len(y), n + len(y)
    return tot / max(n, 1)


@torch.no_grad()
def predict(net, model, dl, device, amp, feats=False):
    train_mode(model, False)
    out = []
    for x, _ in dl:
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            r = (model.features if feats else net)(x.to(device, non_blocking=True)).float().cpu()
        out.append(r if feats else r[:, 1].numpy())
    return torch.cat(out) if feats else np.concatenate(out)


def fit(net, model, tl, vl, crit, a, device, amp, ckpt, log):
    opt = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=a.lr, weight_decay=a.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.1, patience=5)
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    best, bad, hist = float("inf"), 0, []
    for ep in range(1, a.epochs + 1):
        t0 = time.time()
        tr = run(net, model, tl, crit, device, amp, opt, scaler)
        va = run(net, model, vl, crit, device, amp)
        sched.step(va)
        hist.append({"epoch": ep, "train_loss": tr, "val_loss": va})
        if va < best - 1e-4:
            best, bad = va, 0
            torch.save({k: v.detach().cpu() for k, v in model.named_parameters() if v.requires_grad}, ckpt)
        else:
            bad += 1
        log(f"  эпоха {ep:3d}: train {tr:.4f}  val {va:.4f}  {time.time() - t0:.0f} с{' *' if bad == 0 else ''}")
        if bad >= a.patience:
            log("  ранняя остановка")
            break
    model.load_state_dict(torch.load(ckpt, weights_only=True), strict=False)
    return hist


def youden(y, p):
    from sklearn.metrics import roc_curve
    if len(np.unique(y)) < 2:
        return 0.5
    fpr, tpr, thr = roc_curve(y, p)
    return float(thr[1:][np.argmax((tpr - fpr)[1:])])


def train_fold(fold, frac, caches, a, device, run_dir, log):
    tag = f"{fold['name']}_frac{round(frac * 100)}"
    out = run_dir / tag
    if (out / "preds.npz").is_file() and not a.force:
        return log(f"{tag}: уже посчитан")
    out.mkdir(parents=True, exist_ok=True)
    names = list(caches)
    cl = [caches[n] for n in names]
    lab = lambda it: np.array([cl[c].labels[i] for c, i in it], np.int64)
    rng = np.random.default_rng(a.seed)
    tr_it, sel = fold["train"], None
    if frac < 1:
        sel, _ = strat(lab(tr_it), frac, rng)
        tr_it = tr_it[sel]
    ytr, yva, yte = lab(tr_it), lab(fold["val"]), lab(fold["test"])
    log(f"{tag}: train {len(ytr)} (приступ {ytr.sum()}), val {len(yva)} ({yva.sum()}), test {len(yte)} ({yte.sum()})")
    if not ytr.sum() or not yva.sum():
        return log(f"{tag}: нет приступов в train/val — пропуск")
    norm = None if a.norm == "window" else global_norm(fold["norm"])
    ds = lambda it: Windows(cl, it, norm)
    cw = (len(ytr) / (2 * np.maximum(np.bincount(ytr, minlength=2), 1))).tolist() if a.balancing == "class_weights" else None
    crit = SSWCE(weights=cw)
    model = BENDR(a.feature)
    model.load_pretrained(a.weights_dir)
    n_tr = set_freeze(model, a.freeze, a.unfreeze_last)
    model.to(device)
    amp = device.type == "cuda" and not a.no_amp
    log(f"{tag}: режим {a.freeze}, обучаемых параметров {n_tr:,}, AMP {amp}")
    ckpt = out / "best.pt"
    if a.freeze == "backbone":   # признаки BENDR один раз на фолд, дальше учится только голова
        if "feats" not in fold:
            t0 = time.time()
            fold["feats"] = [predict(None, model, DataLoader(ds(fold[k]), a.batch_size, num_workers=a.workers),
                                     device, amp, feats=True) for k in ("train", "val", "test")]
            log(f"{fold['name']}: признаки за {time.time() - t0:.0f} с")
        ftr, fva, fte = fold["feats"]
        if sel is not None:
            ftr = ftr[torch.as_tensor(sel)]
        tl = loader(TensorDataset(ftr, torch.as_tensor(ytr)), ytr, a, True, device, 0)
        vl = DataLoader(TensorDataset(fva, torch.as_tensor(yva)), a.batch_size)
        tel = DataLoader(TensorDataset(fte, torch.as_tensor(yte)), a.batch_size)
        hist = fit(model.head, model, tl, vl, crit, a, device, False, ckpt, log)
        pva, pte = predict(model.head, model, vl, device, False), predict(model.head, model, tel, device, False)
    else:
        tl = loader(ds(tr_it), ytr, a, True, device, a.workers)
        vl, tel = (loader(ds(fold[k]), None, a, False, device, a.workers) for k in ("val", "test"))
        hist = fit(model, model, tl, vl, crit, a, device, amp, ckpt, log)
        pva, pte = predict(model, model, vl, device, amp), predict(model, model, tel, device, amp)
    thr = youden(yva, pva) if a.threshold == "youden" else 0.5
    tc, ti = cl[int(fold["test"][0, 0])], fold["test"][:, 1]
    np.savez_compressed(out / "preds.npz", p=pte, y=yte, file_idx=tc.file_idx[ti], starts=tc.starts[ti])
    events = {int(i): tc.files[i]["seizures"] for i in np.unique(tc.file_idx[ti])}
    info = {"fold": fold["name"], "fraction": frac, "freeze": a.freeze, "threshold": thr,
            "val_auc": auc(yva, pva), "events": events, "history": hist}
    (out / "fold.json").write_text(json.dumps(info, ensure_ascii=False, indent=1))
    log(f"{tag}: порог {thr:.3f}, val AUC {info['val_auc']}, test AUC {auc(yte, pte)}")


# ----------------------------------------------------- оценка и пост-обработка

def auc(y, p):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else None


def vote(pred, w):
    """Голосование большинством в нечётном окне w (края дополняются повтором)."""
    if w == 1:
        return pred
    pad = np.pad(pred, (w // 2, w // 2), mode="edge")
    return (np.convolve(pad, np.ones(w, int), "valid") > w // 2).astype(int)


def metrics(y, p, fid, st, events, thr, w):
    pred = (p >= thr).astype(int)
    out = np.empty_like(pred)
    groups = hit = total = 0
    for f in np.unique(fid):   # голосование и группировка FP внутри файла
        m = fid == f
        pf = vote(pred[m], w)
        out[m] = pf
        fp = (pf == 1) & (y[m] == 0)
        groups += int(fp[0]) + int((fp[1:] & ~fp[:-1]).sum())
        for a0, b0 in events.get(int(f), []):
            ov = (st[m] < b0) & (st[m] + P.WINDOW_SEC > a0)
            if ov.any():
                total, hit = total + 1, hit + int(pf[ov].any())
    tp, tn = int(((out == 1) & (y == 1)).sum()), int(((out == 0) & (y == 0)).sum())
    fp, fn = int(((out == 1) & (y == 0)).sum()), int(((out == 0) & (y == 1)).sum())
    spec = tn / (tn + fp) if tn + fp else None
    hours = (tn + fp) * P.WINDOW_SEC / 3600
    return {"vote_width": w, "stage": "до" if w == 1 else "после", "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "sensitivity": tp / (tp + fn) if tp + fn else None, "specificity": spec,
            "fp_per_hour": None if spec is None else (1 - spec) * 3600 / P.WINDOW_SEC,  # формула диссертации
            "grouped_fp_per_hour": groups / hours if hours else None,
            "event_sensitivity": hit / total if total else None, "n_events": total}


def evaluate(run_dir, widths, thr_override=None):
    rows = []
    for d in sorted(x for x in run_dir.iterdir() if (x / "preds.npz").is_file()):
        info = json.loads((d / "fold.json").read_text())
        z = np.load(d / "preds.npz")
        ev = {int(k): v for k, v in info["events"].items()}
        thr = info["threshold"] if thr_override is None else thr_override
        for w in widths:
            rows.append({"fold": info["fold"], "fraction": info["fraction"], "auc": auc(z["y"], z["p"]),
                         "threshold": thr, **metrics(z["y"], z["p"], z["file_idx"], z["starts"], ev, thr, w)})
    if not rows:
        return print("Нет предсказаний для оценки.")
    df = pd.DataFrame(rows)
    df.to_csv(run_dir / "metrics_all.csv", index=False)
    cols = ["auc", "event_sensitivity", "sensitivity", "specificity", "fp_per_hour", "grouped_fp_per_hour"]
    s = df.groupby(["fraction", "vote_width", "stage"])[cols].mean().round(4).reset_index()
    s.insert(3, "n_folds", df.groupby(["fraction", "vote_width"]).size().values)
    s.to_csv(run_dir / "summary.csv", index=False)
    (run_dir / "metrics.json").write_text(json.dumps({"per_fold": rows, "summary": s.to_dict("records")},
                                                     ensure_ascii=False, indent=1, default=float))
    print(s.to_string(index=False))
    print(f"\nРезультаты: {run_dir}")


# ------------------------------------------------------------------- запуск

def main():
    ap = argparse.ArgumentParser(description="Дообучение BENDR на CHB-MIT + оценка (значения по умолчанию — из диссертации).")
    P.add_path_args(ap)
    ap.add_argument("--results-dir", default=str(P.ROOT / "results"))
    ap.add_argument("--run-name", default=None, help="по умолчанию <схема>_<freeze>")
    ap.add_argument("--scheme", choices=["lopo", "patient_wise"], default="lopo")
    ap.add_argument("--test-patients", nargs="+", help="какие фолды считать (по умолчанию все)")
    ap.add_argument("--freeze", choices=["full", "encoder", "partial", "backbone"], default="full")
    ap.add_argument("--unfreeze-last", type=int, default=2, help="для partial (выбрано автором)")
    ap.add_argument("--fractions", nargs="+", type=float, default=[1.0], help="например 0.25 0.5 1.0")
    ap.add_argument("--balancing", choices=["sampler", "class_weights", "none"], default="sampler")
    ap.add_argument("--norm", choices=["global", "window"], default="global")
    ap.add_argument("--feature", choices=["ctx_first", "ctx_mean", "enc_mean"], default="ctx_first")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--patience", type=int, default=15, help="ранняя остановка по val_loss")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--test-fraction", type=float, default=0.3, help="для patient_wise (выбрано автором)")
    ap.add_argument("--max-bg", type=int, default=None, help="урезать фон в обучающих файлах до N окон")
    ap.add_argument("--samples-per-epoch", type=int, default=None, help="укоротить эпоху (для sampler)")
    ap.add_argument("--threshold", choices=["youden", "argmax"], default="youden")
    ap.add_argument("--vote-widths", nargs="+", type=int, default=[1, 3, 5, 7], help="1 = без пост-обработки")
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--workers", type=int, default=4, help="процессов DataLoader")
    ap.add_argument("--threads", type=int, default=None, help="потоков torch на CPU")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true", help="пересчитать готовые фолды")
    ap.add_argument("--eval-only", action="store_true", help="только пересчитать метрики")
    a = ap.parse_args()
    if any(w % 2 == 0 for w in a.vote_widths):
        sys.exit("Окна голосования должны быть нечётными.")

    run_dir = Path(a.results_dir) / (a.run_name or f"{a.scheme}_{a.freeze}")
    run_dir.mkdir(parents=True, exist_ok=True)
    if a.eval_only:
        return evaluate(run_dir, a.vote_widths)

    if a.device == "cuda" and not torch.cuda.is_available():
        print("CUDA недоступна — продолжаю на CPU.")
    device = torch.device("cuda" if a.device != "cpu" and torch.cuda.is_available() else "cpu")
    if a.threads:
        torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    P.prepare_all(a.patients, a.raw_dir, a.cache_dir, a.weights_dir, a.prep_workers, not a.no_download, a.on_missing)
    caches = {p: P.PatientCache(a.cache_dir, p) for p in a.patients if (Path(a.cache_dir) / p / "meta.json").is_file()}
    logf = (run_dir / "train.log").open("a", encoding="utf-8")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    (run_dir / "args.json").write_text(json.dumps(vars(a), ensure_ascii=False, indent=1))
    log(f"=== {time.strftime('%Y-%m-%d %H:%M')} | {device} "
        f"{torch.cuda.get_device_name(0) if device.type == 'cuda' else ''} | пациентов {len(caches)} | {run_dir}")
    for fold in make_folds(a, caches, np.random.default_rng(a.seed)):
        for frac in a.fractions:
            train_fold(fold, frac, caches, a, device, run_dir, log)
        fold.pop("feats", None)
    evaluate(run_dir, a.vote_widths)


if __name__ == "__main__":
    main()
