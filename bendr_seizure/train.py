# -*- coding: utf-8 -*-
"""
Назначение: цикл дообучения BENDR, ранняя остановка и схема LOPO.
Дата: 23.11.2026

Оптимизатор Adam, планировщик ReduceLROnPlateau и ранняя остановка по
val_loss — как в исходном репозитории и в диссертации (lr = 1e-4,
weight decay = 0,01). На CPU без GPU число эпох ограничено автором.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit

from .dataset import make_loader, take_fraction
from .loss import SensitivitySpecificityWeightedCrossEntropy
from .model import BENDRClassifier


def pick_device(cfg, override=None):
    """Выбор устройства: auto (по умолчанию), cpu или cuda.

    Значение берётся из override, затем из cfg['device'].
    Старый формат {'prefer_cuda': true/false} тоже принимается.
    """
    name = override if override is not None else cfg.get("device", "auto")
    if isinstance(name, dict):
        prefer = bool(name.get("prefer_cuda", True))
        name = "auto" if prefer else "cpu"
    name = str(name or "auto").strip().lower()
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("В конфиге или аргументах указано cuda, но GPU недоступен.")
        return torch.device("cuda")
    if name == "cpu":
        return torch.device("cpu")
    # auto и любые прочие значения: CUDA, если есть, иначе CPU
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def batch_size(cfg, device):
    tr = cfg["training"]
    if device.type == "cuda":
        return int(tr["batch_size"])
    return int(tr["batch_size_cpu"])


class EarlyStopping:
    def __init__(self, patience, min_delta):
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.best = None
        self.bad = 0
        self.stop = False

    def step(self, value):
        if self.best is None or value < self.best - self.min_delta:
            self.best = value
            self.bad = 0
        else:
            self.bad += 1
        self.stop = self.bad >= self.patience
        return self.stop


class Trainer:
    def __init__(self, model, cfg, device):
        self.model = model.to(device)
        self.device = device
        tr = cfg["training"]
        self.criterion = SensitivitySpecificityWeightedCrossEntropy.from_config(cfg)
        params = [par for par in self.model.parameters() if par.requires_grad]
        if not params:
            raise RuntimeError("Нет обучаемых параметров.")
        self.opt = Adam(params, lr=float(tr["lr"]), weight_decay=float(tr["weight_decay"]))
        self.sched = ReduceLROnPlateau(
            self.opt, mode=str(tr.get("scheduler_mode", "min")),
            factor=float(tr["scheduler_factor"]),
            patience=int(tr["scheduler_patience"]),
        )
        self.stopper = EarlyStopping(tr["early_stopping_patience"], tr["early_stopping_min_delta"])
        self.ckpt_extra = {}

    def _run(self, loader, train):
        self.model.train(train)
        total, seen = 0.0, 0
        with torch.set_grad_enabled(train):
            for x, y in loader:
                x = x.to(self.device)
                y = y.to(self.device).long()
                if train:
                    self.opt.zero_grad()
                loss = self.criterion(self.model(x), y)
                if train:
                    loss.backward()
                    self.opt.step()
                bs = int(y.shape[0])
                total += float(loss.detach()) * bs
                seen += bs
        if seen == 0:
            raise RuntimeError("Пустой загрузчик.")
        return total / seen

    def fit(self, train_loader, val_loader, epochs, ckpt_path):
        history = []
        ckpt_path = Path(ckpt_path)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        for epoch in range(1, int(epochs) + 1):
            tr = self._run(train_loader, True)
            va = self._run(val_loader, False)
            self.sched.step(va)
            improved = self.stopper.bad == 0 and (self.stopper.best is None or va <= self.stopper.best)
            # step ещё не вызван; вызываем после сравнения через stopper.step
            stop = self.stopper.step(va)
            if self.stopper.bad == 0:
                payload = {"model": self.model.state_dict(), "val_loss": va, "epoch": epoch}
                if getattr(self, "ckpt_extra", None):
                    payload.update(self.ckpt_extra)
                torch.save(payload, ckpt_path)
            history.append({"epoch": epoch, "train_loss": tr, "val_loss": va})
            print(f"  Эпоха {epoch}: train {tr:.4f}, val {va:.4f}", flush=True)
            if stop:
                print("  Ранняя остановка.", flush=True)
                break
        if ckpt_path.is_file():
            state = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(state["model"])
        return history

    @torch.no_grad()
    def predict(self, windows, bs):
        self.model.eval()
        probs = []
        x_all = torch.from_numpy(np.ascontiguousarray(windows, dtype=np.float32))
        for i in range(0, len(x_all), bs):
            batch = x_all[i:i + bs].to(self.device)
            probs.append(self.model(batch).cpu().numpy())
        return np.concatenate(probs, 0)



def _metrics_from_probs(probs, y):
    pred = probs.argmax(dim=-1)
    y = y.long()
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    sen = tp / (tp + fn) if (tp + fn) else None
    spe = tn / (tn + fp) if (tn + fp) else None
    p1 = float(probs[:, 1].mean())
    return sen, spe, p1, {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def freeze_backbone(model, cfg):
    """Заморозить энкодер и/или контекстуализатор. Выбор автора для CPU."""
    tr = cfg.get("training") or {}
    n_enc = n_ctx = 0
    if tr.get("freeze_encoder"):
        for par in model.encoder.parameters():
            par.requires_grad = False
            n_enc += par.numel()
        model.encoder.eval()
    if tr.get("freeze_contextualizer"):
        for par in model.contextualizer.parameters():
            par.requires_grad = False
            n_ctx += par.numel()
        model.contextualizer.eval()
    trainable = sum(par.numel() for par in model.parameters() if par.requires_grad)
    print(f"Заморожено: энкодер {n_enc} параметров, контекстуализатор {n_ctx}; обучаемых {trainable}.", flush=True)
    return trainable


@torch.no_grad()
def extract_features(model, windows, bs, device, log_every=40, variant="ctx_first"):
    """Прямой проход замороженного BENDR: один из вариантов признаков 512-d.

    variant: ctx_first, ctx_last, ctx_mean или enc_mean (см. features.VARIANTS).
    По умолчанию — первый токен, как в официальной голове и в ранних прогонах.
    """
    from .features import VARIANTS, extract_variants
    if variant not in VARIANTS:
        raise ValueError(f"Неизвестный feature_variant={variant!r}. Допустимы: {VARIANTS}.")
    bags = extract_variants(model, windows, bs, device, log_every=log_every)
    return bags[variant]


def fit_head(model, cfg, device, feat_tr, y_tr, feat_va, y_va, epochs, ckpt_path, extra=None):
    """Дообучение только головы на заранее посчитанных признаках (эквивалент freeze)."""
    from torch.utils.data import DataLoader, TensorDataset
    from .dataset import make_loader

    tr = cfg["training"]
    bs = int(tr.get("batch_size_cpu") if device.type != "cuda" else tr["batch_size"])
    # взвешенная выборка — как в репозитории
    y_tr = np.asarray(y_tr, dtype=np.int64)
    y_va = np.asarray(y_va, dtype=np.int64)
    loader_tr = make_loader(feat_tr, y_tr, bs, weighted=True, seed=int(cfg["seed"]))
    loader_va = make_loader(feat_va, y_va, bs, shuffle=False, seed=int(cfg["seed"]))

    criterion = SensitivitySpecificityWeightedCrossEntropy.from_config(cfg)
    opt = Adam(model.head.parameters(), lr=float(tr["lr"]), weight_decay=float(tr["weight_decay"]))
    sched = ReduceLROnPlateau(
        opt, mode=str(tr.get("scheduler_mode", "min")),
        factor=float(tr["scheduler_factor"]),
        patience=int(tr["scheduler_patience"]),
    )
    stopper = EarlyStopping(tr["early_stopping_patience"], tr["early_stopping_min_delta"])
    ckpt_path = Path(ckpt_path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    history = []
    model.head.to(device)

    def run(loader, train):
        model.head.train(train)
        total, seen = 0.0, 0
        all_p, all_y = [], []
        with torch.set_grad_enabled(train):
            for x, y in loader:
                x = x.to(device)
                y = y.to(device).long()
                if train:
                    opt.zero_grad()
                probs = model.head(x)
                loss = criterion(probs, y)
                if train:
                    loss.backward()
                    opt.step()
                total += float(loss.detach()) * int(y.shape[0])
                seen += int(y.shape[0])
                all_p.append(probs.detach())
                all_y.append(y.detach())
        probs = torch.cat(all_p, 0)
        yt = torch.cat(all_y, 0)
        sen, spe, p1, _ = _metrics_from_probs(probs, yt)
        return total / max(seen, 1), sen, spe, p1

    for epoch in range(1, int(epochs) + 1):
        tr_loss, tr_s, tr_p, tr_p1 = run(loader_tr, True)
        va_loss, va_s, va_p, va_p1 = run(loader_va, False)
        sched.step(va_loss)
        stop = stopper.step(va_loss)
        if stopper.bad == 0:
            # «head» — формат зонда (scripts_final после 23.11); «model» оставляем
            # для совместимости с полным state_dict и старыми чекпоинтами.
            payload = {
                "head": model.head.state_dict(),
                "model": model.state_dict(),
                "val_loss": va_loss,
                "epoch": epoch,
            }
            if extra:
                payload.update(extra)
            torch.save(payload, ckpt_path)
        history.append({
            "epoch": epoch, "train_loss": tr_loss, "val_loss": va_loss,
            "train_sens": tr_s, "train_spec": tr_p, "val_sens": va_s, "val_spec": va_p,
            "train_p1": tr_p1, "val_p1": va_p1,
        })
        def fmt(v):
            return "n/a" if v is None else f"{v:.3f}"
        print(
            f"  Эпоха {epoch}: train {tr_loss:.4f} (sens {fmt(tr_s)}, spec {fmt(tr_p)}, p1 {tr_p1:.3f}) | "
            f"val {va_loss:.4f} (sens {fmt(va_s)}, spec {fmt(va_p)}, p1 {va_p1:.3f})",
            flush=True,
        )
        if stop:
            print("  Ранняя остановка.", flush=True)
            break
    if ckpt_path.is_file():
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        if "head" in state:
            model.head.load_state_dict(state["head"])
        else:
            model.load_state_dict(state["model"])
    return history


def split_train_val(labels, val_fraction, seed):
    n = len(labels)
    counts = np.bincount(labels.astype(int), minlength=2)
    if counts.min() >= 2:
        sp = StratifiedShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
        tr, va = next(sp.split(np.zeros(n), labels))
    else:
        sp = ShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
        tr, va = next(sp.split(np.zeros(n)))
    return tr, va


def build_model(cfg, device):
    model = BENDRClassifier(cfg)
    wdir = cfg["paths"]["weights_dir"]
    names = cfg.get("weights") or {}
    report = model.load_pretrained(
        wdir,
        names.get("encoder_filename", "encoder.pt"),
        names.get("contextualizer_filename", "contextualizer.pt"),
    )
    model = model.to(device)
    freeze_backbone(model, cfg)
    return model, report
