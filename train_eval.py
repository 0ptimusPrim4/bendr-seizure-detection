# -*- coding: utf-8 -*-
# ====================== ПУТИ (при необходимости поменяйте здесь) ======================
# Папка со скриптом и весами BENDR v0.1-alpha (encoder.pt, contextualizer.pt); сюда же results\
SCRIPTS_DIR = r"C:\Users\abuzo\Desktop\МАГИСТРАТУРА\МАГИСТЕРСКАЯ\3 семестр\scripts"
# Папка CHB-MIT: внутри chb01, chb02, ... с *.edf и chbXX-summary.txt
DATA_DIR = r"C:\Users\abuzo\Desktop\МАГИСТРАТУРА\МАГИСТЕРСКАЯ\3 семестр\data\chb-mit-scalp-eeg-database-1.0.0"
# =====================================================================================
r"""
Дообучение BENDR на CHB-MIT + оценка и пост-обработка, одним скриптом:

  python train_eval.py

Данные и веса кладутся вручную (скрипт ничего не скачивает):
  DATA_DIR\chb01\*.edf + chb01-summary.txt, ..., chb24\...
  SCRIPTS_DIR\encoder.pt, SCRIPTS_DIR\contextualizer.pt  (BENDR v0.1-alpha)
Если этих папок нет (другой ПК), веса и results ищутся рядом со скриптом, данные — в data/chbmit.
EDF читаются сами: 20 каналов 10-20, 256 Гц, Баттерворт 0,5–50 Гц (sosfiltfilt, порядок 5),
окна 4 с / 1024 отсчёта, 15 мин постиктала исключаются; окна кэшируются в --cache-dir.
Режимы --freeze: full — учится всё (диссертация); encoder — заморожен свёрточный энкодер;
partial — последние --unfreeze-last слоёв трансформера и голова; backbone — только голова.
Доли 25/50/100 % (--fractions) — объём обучающей выборки.
Метрики до/после пост-обработки (голосование в нечётном окне, группировка FP) — <out>/<run>/summary.csv.
"""

import argparse
import hashlib
import importlib.util
import json
import re
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import os

ROOT = Path(__file__).resolve().parent
# Фоллбек, если Windows-пути из начала файла на этом ПК не существуют
_SCRIPTS = Path(SCRIPTS_DIR) if Path(SCRIPTS_DIR).is_dir() else ROOT
_DATA = Path(DATA_DIR) if Path(DATA_DIR).is_dir() else ROOT / "data" / "chbmit"

# Параметры предобработки (диссертация Benfenati / исходный configs/default.yaml)
SFREQ = 256.0
WINDOW_SEC = 4.0
POSTICTAL_MIN = 15.0
FILTER = (5, 0.5, 50.0)            # порядок, нижняя и верхняя частоты
DROP = ["FT9-FT10", "FT10-T8", "T7-FT9"]
# Слоты BENDR (19 электродов 10-20 + SCALE) ← биполярные отведения CHB-MIT в порядке сводки chb01.
# Это порядок слотов, а не геометрия: таблицы пар в диссертации нет.
MAPPING = {
    "FP1": "FP1-F7", "FP2": "F7-T7", "F7": "T7-P7", "F3": "P7-O1", "FZ": "FP1-F3",
    "F4": "F3-C3", "F8": "C3-P3", "T7": "P3-O1", "C3": "FP2-F4", "CZ": "F4-C4",
    "C4": "C4-P4", "T8": "P4-O2", "T5": "FP2-F8", "P3": "F8-T8", "PZ": "T8-P8",
    "P4": "P8-O2", "T6": "FZ-CZ", "O1": "CZ-PZ", "O2": "P7-T7",
}
SIGNATURE = hashlib.sha1(json.dumps([SFREQ, WINDOW_SEC, POSTICTAL_MIN, FILTER, DROP, MAPPING]).encode()).hexdigest()[:12]


def require(*modules):
    """Понятная ошибка, если не установлены зависимости."""
    missing = [m for m in modules if importlib.util.find_spec(m) is None]
    if missing:
        sys.exit(f"Не установлены пакеты: {', '.join(missing)}.\n"
                 f"Выполните: pip install -r requirements.txt (torch — с сайта pytorch.org под вашу CUDA).")


require("numpy", "scipy", "mne", "torch", "sklearn", "pandas")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader, Dataset, TensorDataset, WeightedRandomSampler  # noqa: E402

warnings.filterwarnings("ignore", message=".*weight_norm.*")


# ------------------------------------------------------- чтение CHB-MIT

def parse_summary(path):
    """chbXX-summary.txt → список (имя файла, [(начало, конец), ...])."""
    out, cur, start = [], None, None
    for line in Path(path).read_text(encoding="utf-8-sig", errors="replace").splitlines():
        line = line.strip()
        if m := re.match(r"^File Name:\s*(.+?)\s*$", line, re.I):
            cur = (m.group(1), [])
            out.append(cur)
        elif cur and (m := re.match(r"^Seizure(?:\s+\d+)?\s+Start Time:\s*([\d.]+)", line, re.I)):
            start = float(m.group(1))
        elif cur and start is not None and (m := re.match(r"^Seizure(?:\s+\d+)?\s+End Time:\s*([\d.]+)", line, re.I)):
            cur[1].append((start, float(m.group(1))))
            start = None
    return out


def edf_labels(path):
    # подписи из заголовка EDF: mne переименовывает дубликаты (T8-P8-0, T8-P8-1)
    with open(path, "rb") as fh:
        n = int(fh.read(256)[252:256].decode("ascii", "ignore").strip() or 0)
        raw = fh.read(n * 16)
    return [raw[i * 16:(i + 1) * 16].decode("ascii", "replace").strip().upper() for i in range(n)]


def load_file(path, seizures, on_missing="skip"):
    """Один EDF → окна [N, 20, 1024], метки, начала окон (с). None — файл пропущен."""
    import mne
    from scipy.signal import butter, sosfiltfilt
    raw = mne.io.read_raw_edf(str(path), preload=True, verbose="ERROR")
    if abs(raw.info["sfreq"] - SFREQ) > 1e-3:
        raw.resample(SFREQ, verbose="ERROR")
    data = raw.get_data().astype(np.float32)
    names = edf_labels(path)
    if len(names) != len(data):
        names = [c.upper() for c in raw.ch_names]
    idx = {}
    for i, n in enumerate(names):
        if n and n not in DROP and n not in idx:   # второй T8-P8 отбрасывается
            idx[n] = i
    absent = [src for src in MAPPING.values() if src not in idx]
    if absent and on_missing != "zero":
        return None, absent
    eeg = np.stack([data[idx[s]] if s in idx else np.zeros(data.shape[1], np.float32) for s in MAPPING.values()])
    scale = np.abs(eeg).mean(0, keepdims=True)  # SCALE — относительная амплитуда (выбрано автором)
    x = np.concatenate([eeg, scale])
    order, lo, hi = FILTER
    sos = butter(order, [lo, hi], btype="bandpass", fs=SFREQ, output="sos")
    x = sosfiltfilt(sos, x.astype(np.float64), axis=-1).astype(np.float32)

    win, n = int(WINDOW_SEC * SFREQ), x.shape[1]
    sz = [(int(a * SFREQ), min(n, int(np.ceil(b * SFREQ)))) for a, b in seizures]
    post = [(b, min(n, b + int(POSTICTAL_MIN * 60 * SFREQ))) for _, b in sz]
    hit = lambda s, e, rr: any(s < b and e > a for a, b in rr)
    starts, labels = [], []
    for s in range(0, n - win + 1, win):
        if hit(s, s + win, sz):
            labels.append(1)
        elif hit(s, s + win, post):
            continue
        else:
            labels.append(0)
        starts.append(s)
    if not starts:
        return None, []
    w = np.stack([x[:, s:s + win] for s in starts])
    return (w, np.array(labels, np.int8), np.array(starts) / SFREQ), []


def prepare_patient(patient, raw_dir, cache_dir, on_missing="skip"):
    """Строит кэш пациента; повторный запуск ничего не пересчитывает."""
    out = Path(cache_dir) / patient
    meta_path = out / "meta.json"
    if meta_path.is_file() and json.loads(meta_path.read_text(encoding="utf-8"))["signature"] == SIGNATURE:
        return f"{patient}: кэш готов"
    raw_dir = Path(raw_dir)
    summary = raw_dir / patient / f"{patient}-summary.txt"
    if not summary.is_file():
        return f"{patient}: нет {summary}, пропуск"
    out.mkdir(parents=True, exist_ok=True)
    files, skipped, t0 = [], [], time.time()
    for name, seizures in parse_summary(summary):
        edf = raw_dir / patient / name
        if not edf.is_file():
            continue
        res, absent = load_file(edf, seizures, on_missing)
        if res is None:
            if absent:
                skipped.append({"file": name, "missing": absent})
            continue
        w, y, st = res
        stem = edf.stem
        np.save(out / f"{stem}_windows.npy", w)
        np.save(out / f"{stem}_labels.npy", y)
        np.save(out / f"{stem}_starts.npy", st)
        w64 = w.astype(np.float64)
        files.append({"file": name, "stem": stem, "n_windows": len(y), "n_seizure": int(y.sum()),
                      "seizures": seizures, "ch_sum": w64.sum((0, 2)).tolist(),
                      "ch_sumsq": (w64 ** 2).sum((0, 2)).tolist(), "ch_count": int(w.shape[0] * w.shape[2])})
    meta = {"patient": patient, "signature": SIGNATURE, "files": files, "skipped_files": skipped,
            "n_windows": sum(f["n_windows"] for f in files), "n_seizure": sum(f["n_seizure"] for f in files)}
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    return (f"{patient}: {len(files)} файлов, {meta['n_windows']} окон (приступ {meta['n_seizure']}), "
            f"пропущено {len(skipped)} файлов с неполным монтажом, {time.time() - t0:.0f} с")


class PatientCache:
    """Окна пациента из кэша; сами окна читаются через memmap по требованию."""

    def __init__(self, cache_dir, patient):
        self.dir = Path(cache_dir) / patient
        self.meta = json.loads((self.dir / "meta.json").read_text(encoding="utf-8"))
        self.files = self.meta["files"]
        self._mm = {}
        self.labels = np.concatenate([np.load(self.dir / f"{f['stem']}_labels.npy") for f in self.files]).astype(np.int64)
        self.starts = np.concatenate([np.load(self.dir / f"{f['stem']}_starts.npy") for f in self.files])
        self.file_idx = np.concatenate([np.full(f["n_windows"], i) for i, f in enumerate(self.files)])
        self.local = np.concatenate([np.arange(f["n_windows"]) for f in self.files])

    def __len__(self):
        return len(self.labels)

    def __getstate__(self):  # в процессы DataLoader memmap не передаём
        return {**self.__dict__, "_mm": {}}

    def window(self, i):
        f = int(self.file_idx[i])
        if f not in self._mm:
            self._mm[f] = np.load(self.dir / f"{self.files[f]['stem']}_windows.npy", mmap_mode="r")
        return np.asarray(self._mm[f][self.local[i]], np.float32)

    def channel_stats(self, file_ids=None):
        ids = range(len(self.files)) if file_ids is None else file_ids
        return (sum(np.array(self.files[i]["ch_sum"]) for i in ids),
                sum(np.array(self.files[i]["ch_sumsq"]) for i in ids),
                sum(self.files[i]["ch_count"] for i in ids))


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

    def load_pretrained(self, enc_path, ctx_path):
        for name, mod, path in (("encoder", self.encoder, enc_path), ("contextualizer", self.contextualizer, ctx_path)):
            miss, unexp = mod.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=False)
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
    model.load_pretrained(a.encoder, a.contextualizer)
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
    (out / "fold.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
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
            ov = (st[m] < b0) & (st[m] + WINDOW_SEC > a0)
            if ov.any():
                total, hit = total + 1, hit + int(pf[ov].any())
    tp, tn = int(((out == 1) & (y == 1)).sum()), int(((out == 0) & (y == 0)).sum())
    fp, fn = int(((out == 1) & (y == 0)).sum()), int(((out == 0) & (y == 1)).sum())
    spec = tn / (tn + fp) if tn + fp else None
    hours = (tn + fp) * WINDOW_SEC / 3600
    return {"vote_width": w, "stage": "до" if w == 1 else "после", "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "sensitivity": tp / (tp + fn) if tp + fn else None, "specificity": spec,
            "fp_per_hour": None if spec is None else (1 - spec) * 3600 / WINDOW_SEC,  # формула диссертации
            "grouped_fp_per_hour": groups / hours if hours else None,
            "event_sensitivity": hit / total if total else None, "n_events": total}


def evaluate(run_dir, widths, thr_override=None):
    rows = []
    for d in sorted(x for x in run_dir.iterdir() if (x / "preds.npz").is_file()):
        info = json.loads((d / "fold.json").read_text(encoding="utf-8"))
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
                                                     ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(s.to_string(index=False))
    print(f"\nРезультаты: {run_dir}")


# ------------------------------------------------------------------- запуск

def check_inputs(a):
    """Проверка путей до старта: понятная ошибка, куда положить файлы."""
    errors = []
    data = Path(a.data_dir)
    if not data.is_dir():
        errors.append(f"Нет папки с CHB-MIT: {data}\n  Положите туда папки chb01 … chb24 с файлами *.edf и chbXX-summary.txt "
                      f"(https://physionet.org/content/chbmit/1.0.0/) или поправьте DATA_DIR в начале train_eval.py / флаг --data-dir.")
    else:
        found = sorted(p.name for p in data.glob("chb*") if (p / f"{p.name}-summary.txt").is_file() and any(p.glob("*.edf")))
        a.patients = a.patients or found
        missing = [p for p in a.patients if p not in found]
        if missing:
            errors.append(f"В {data} нет данных пациентов: {', '.join(missing)}\n"
                          f"  Ожидается {data}/chbXX/chbXX-summary.txt и {data}/chbXX/*.edf.")
        elif len(a.patients) < 2 and a.scheme == "lopo":
            errors.append(f"Для LOPO нужно минимум 2 пациента, найдено: {a.patients or 'ни одного'} в {data}.")
    for flag, path in (("--encoder", a.encoder), ("--contextualizer", a.contextualizer)):
        if not Path(path).is_file():
            errors.append(f"Нет весов BENDR: {path}\n  Положите файл {Path(path).name} (релиз v0.1-alpha, "
                          f"https://github.com/SPOClab-ca/BENDR/releases/tag/v0.1-alpha) по этому пути (папка SCRIPTS_DIR в начале train_eval.py) или укажите {flag}.")
    if errors:
        sys.exit("ОШИБКА:\n" + "\n".join(errors))


def build_cache(a):
    """Читает EDF и кэширует окна (один раз; при повторном запуске берётся готовый кэш)."""
    Path(a.cache_dir).mkdir(parents=True, exist_ok=True)
    todo = [p for p in a.patients if not (Path(a.cache_dir) / p / "meta.json").is_file()
            or json.loads((Path(a.cache_dir) / p / "meta.json").read_text(encoding="utf-8"))["signature"] != SIGNATURE]
    if not todo:
        return
    print(f"Предобработка EDF: {', '.join(todo)} → {a.cache_dir}")
    if a.prep_workers > 1:
        with ProcessPoolExecutor(a.prep_workers) as ex:
            for msg in ex.map(prepare_patient, todo, [a.data_dir] * len(todo), [a.cache_dir] * len(todo),
                              [a.on_missing] * len(todo)):
                print(msg, flush=True)
    else:
        for p in todo:
            print(prepare_patient(p, a.data_dir, a.cache_dir, a.on_missing), flush=True)


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="Дообучение BENDR на CHB-MIT + оценка (значения по умолчанию — из диссертации).")
    ap.add_argument("--data-dir", default=str(_DATA), help="папка с chb01…chb24 (*.edf и *-summary.txt)")
    ap.add_argument("--encoder", default=str(_SCRIPTS / "encoder.pt"))
    ap.add_argument("--contextualizer", default=str(_SCRIPTS / "contextualizer.pt"))
    ap.add_argument("--out", default=str(_SCRIPTS / "results"), help="результаты; кэш окон — <out>/cache")
    ap.add_argument("--cache-dir", default=None, help="кэш окон, по умолчанию <out>/cache")
    ap.add_argument("--patients", nargs="+", default=None, help="по умолчанию все chbXX в --data-dir")
    ap.add_argument("--on-missing", choices=["skip", "zero"], default="skip",
                    help="EDF с неполным монтажом: пропустить или заполнить каналы нулями")
    ap.add_argument("--prep-workers", type=int, default=1 if os.name == "nt" else 2, help="процессов чтения EDF")
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
    # на Windows процессы DataLoader запускаются через spawn и часто мешают — по умолчанию 0
    ap.add_argument("--workers", type=int, default=0 if os.name == "nt" else 4, help="процессов DataLoader")
    ap.add_argument("--threads", type=int, default=None, help="потоков torch на CPU")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true", help="пересчитать готовые фолды")
    ap.add_argument("--eval-only", action="store_true", help="только пересчитать метрики")
    a = ap.parse_args()
    if any(w % 2 == 0 for w in a.vote_widths):
        sys.exit("Окна голосования должны быть нечётными.")

    a.cache_dir = a.cache_dir or str(Path(a.out) / "cache")
    if not a.eval_only:
        check_inputs(a)
    run_dir = Path(a.out) / (a.run_name or f"{a.scheme}_{a.freeze}")
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

    build_cache(a)
    caches = {p: PatientCache(a.cache_dir, p) for p in a.patients if (Path(a.cache_dir) / p / "meta.json").is_file()}
    logf = (run_dir / "train.log").open("a", encoding="utf-8")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    (run_dir / "args.json").write_text(json.dumps(vars(a), ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"=== {time.strftime('%Y-%m-%d %H:%M')} | {device} "
        f"{torch.cuda.get_device_name(0) if device.type == 'cuda' else ''} | пациентов {len(caches)} | {run_dir}")
    for fold in make_folds(a, caches, np.random.default_rng(a.seed)):
        for frac in a.fractions:
            train_fold(fold, frac, caches, a, device, run_dir, log)
        fold.pop("feats", None)
    evaluate(run_dir, a.vote_widths)


if __name__ == "__main__":
    main()
