# -*- coding: utf-8 -*-
"""
Общий модуль для prepare_data.py, train.py и evaluate.py (сам не запускается).

Здесь собрано всё, что раньше было разнесено по пакету bendr_seizure/ и scripts/:
  - чтение YAML-конфига и переопределение ключей из командной строки;
  - предобработка CHB-MIT: разбор summary, чтение EDF, маппинг каналов 10-20,
    полосовой Баттерворт 0,5–50 Гц, окна 4 с (1024 отсчёта при 256 Гц);
  - модель BENDR, совместимая с официальными весами v0.1-alpha (SPOClab-ca);
  - функция потерь SSWCE (Benfenati, 2023, ур. 5.9);
  - пост-обработка (голосование большинством, группировка ложных срабатываний)
    и метрики (чувствительность, специфичность, FP/ч, событийная чувствительность, AUC).

Пометка «выбрано автором» в комментариях означает, что значение или приём
не взяты из диссертации Benfenati, а введены в ходе НИР-3.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Конфигурация
# ---------------------------------------------------------------------------

def load_config(path):
    """Читает YAML; относительные пути из paths приводятся к корню репозитория."""
    p = Path(path)
    if not p.is_file():
        p = ROOT / path
    with p.open(encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ValueError(f"Конфиг {p} должен быть словарём YAML.")
    cfg["_config_path"] = str(p.resolve())
    return cfg


def resolve_path(value):
    p = Path(value)
    return p if p.is_absolute() else (ROOT / p).resolve()


def apply_overrides(cfg, pairs):
    """Переопределения вида --set training.lr=1e-4 data.patients=[chb01,chb02]."""
    for item in pairs or []:
        if "=" not in item:
            raise ValueError(f"Ожидалось ключ=значение, получено: {item}")
        key, raw = item.split("=", 1)
        value = yaml.safe_load(raw)
        cur = cfg
        parts = key.strip().split(".")
        for name in parts[:-1]:
            cur = cur.setdefault(name, {})
        cur[parts[-1]] = value
    return cfg


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(name="auto"):
    name = str(name or "auto").lower()
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Указано device=cuda, но CUDA недоступна (проверьте сборку torch).")
    if name == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def preprocessing_signature(cfg):
    """Хэш параметров предобработки: кэш пересчитывается, если они изменились."""
    d = cfg["data"]
    keys = {k: d.get(k) for k in ("sampling_rate_hz", "window_sec", "postictal_exclusion_min",
                                  "filter", "channels")}
    blob = json.dumps(keys, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Предобработка CHB-MIT
# ---------------------------------------------------------------------------

_FILE_RE = re.compile(r"^File Name:\s*(.+?)\s*$", re.I)
_START_RE = re.compile(r"^Seizure(?:\s+\d+)?\s+Start Time:\s*([+-]?\d+(?:\.\d+)?)\s+seconds?\s*$", re.I)
_END_RE = re.compile(r"^Seizure(?:\s+\d+)?\s+End Time:\s*([+-]?\d+(?:\.\d+)?)\s+seconds?\s*$", re.I)


@dataclass
class EdfRecord:
    file_name: str
    seizures: list  # список пар (начало, конец) в секундах


def parse_summary(path):
    """Разбирает chbXX-summary.txt: имена файлов и интервалы приступов."""
    records, current, start = [], None, None
    for line in Path(path).read_text(encoding="utf-8-sig", errors="replace").splitlines():
        line = line.strip()
        m = _FILE_RE.match(line)
        if m:
            current = EdfRecord(m.group(1).strip(), [])
            records.append(current)
            start = None
            continue
        if current is None:
            continue
        m = _START_RE.match(line)
        if m:
            start = float(m.group(1))
            continue
        m = _END_RE.match(line)
        if m and start is not None:
            current.seizures.append((start, float(m.group(1))))
            start = None
    return records


def _edf_labels(path):
    """Подписи каналов прямо из заголовка EDF (mne переименовывает дубликаты)."""
    with open(path, "rb") as fh:
        header = fh.read(256)
        n_sig = int(header[252:256].decode("ascii", "ignore").strip() or "0")
        raw = fh.read(n_sig * 16)
    return [raw[i * 16:(i + 1) * 16].decode("ascii", "replace").replace("\x00", "").strip()
            for i in range(n_sig)]


def load_edf(path, target_sfreq):
    """Чтение EDF через mne; ресемплинг до target_sfreq, если частота другая."""
    import mne
    raw = mne.io.read_raw_edf(str(path), preload=True, verbose="ERROR")
    if abs(float(raw.info["sfreq"]) - target_sfreq) > 1e-3:
        raw.resample(target_sfreq, verbose="ERROR")
    data = np.ascontiguousarray(raw.get_data(), dtype=np.float32)
    names = _edf_labels(path)
    if len(names) != data.shape[0]:
        names = [c.strip() for c in raw.ch_names]
    return data, names, float(raw.info["sfreq"])


def _norm_name(name):
    return " ".join(str(name).strip().upper().split())


class MissingChannels(ValueError):
    pass


def map_channels(data, names, cfg):
    """19 биполярных отведений раскладываются по слотам 10-20 BENDR, 20-й канал — SCALE.

    SCALE считается как относительная амплитуда (среднее модулей 19 каналов) —
    выбрано автором, по аналогии с To1020(include_scale_ch=True) из BENDR.
    """
    ch = cfg["data"]["channels"]
    drop = {_norm_name(n) for n in ch.get("drop", [])}
    available = {}
    for i, name in enumerate(names):
        key = _norm_name(name)
        if not key or key in drop or key in available:   # повторный T8-P8 отбрасывается
            continue
        available[key] = i
    rows, absent = [], []
    for slot in ch["bendr_order"]:
        if slot == "SCALE":
            continue
        src = ch["mapping"][slot]
        idx = available.get(_norm_name(src))
        if idx is None:
            absent.append(f"{slot}←{src}")
            rows.append(np.zeros(data.shape[1], dtype=np.float32))
        else:
            rows.append(data[idx].astype(np.float32))
    if absent and str(ch.get("on_missing", "skip_file")).lower() != "zero":
        raise MissingChannels("нет каналов: " + ", ".join(absent))
    eeg = np.stack(rows, 0)
    scale = np.mean(np.abs(eeg), axis=0, keepdims=True).astype(np.float32)
    return np.concatenate([eeg, scale], 0)


def bandpass(data, sfreq, cfg):
    """Прямой и обратный проход Баттерворта (filtfilt), фаза не сдвигается."""
    from scipy.signal import butter, sosfiltfilt
    f = cfg["data"]["filter"]
    sos = butter(int(f["order"]), [float(f["low_hz"]), float(f["high_hz"])],
                 btype="bandpass", fs=float(sfreq), output="sos")
    return np.ascontiguousarray(sosfiltfilt(sos, data.astype(np.float64), axis=-1), dtype=np.float32)


def make_windows(signal, sfreq, seizures, cfg):
    """Окна без перекрытия. Метка 1 — окно пересекается с приступом.

    Окна постиктального периода (15 мин после приступа) исключаются.
    """
    d = cfg["data"]
    win = int(round(float(d["window_sec"]) * sfreq))
    excl = int(round(float(d["postictal_exclusion_min"]) * 60.0 * sfreq))
    n = signal.shape[1]
    sz = [(max(0, int(np.floor(a * sfreq))), min(n, int(np.ceil(b * sfreq)))) for a, b in seizures]
    sz = [(a, b) for a, b in sz if b > a]
    post = [(b, min(n, b + excl)) for _, b in sz]
    hit = lambda s, e, rr: any(s < y and e > x for x, y in rr)
    starts, labels = [], []
    for s in range(0, n - win + 1, win):
        e = s + win
        if hit(s, e, sz):
            labels.append(1)
        elif hit(s, e, post):
            continue
        else:
            labels.append(0)
        starts.append(s)
    if not starts:
        return np.empty((0, signal.shape[0], win), np.float32), np.empty(0, np.int8), np.empty(0)
    w = np.stack([signal[:, s:s + win] for s in starts], 0)
    return w, np.asarray(labels, np.int8), np.asarray(starts, np.float64) / sfreq


# ---------------------------------------------------------------------------
# Кэш окон (результат prepare_data.py)
# ---------------------------------------------------------------------------

class PatientCache:
    """Окна одного пациента: по .npy на файл EDF (читаются через memmap) и meta.json."""

    def __init__(self, cache_dir, patient):
        self.dir = Path(cache_dir) / patient
        meta_path = self.dir / "meta.json"
        if not meta_path.is_file():
            raise FileNotFoundError(f"Нет кэша {meta_path}. Сначала запустите prepare_data.py.")
        self.meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.patient = patient
        self.files = self.meta["files"]           # список словарей по файлам
        self._mm = {}
        self.labels = np.concatenate([np.load(self.dir / f["labels"]) for f in self.files]).astype(np.int64)
        self.starts = np.concatenate([np.load(self.dir / f["starts"]) for f in self.files])
        self.file_idx = np.concatenate([np.full(f["n_windows"], i, np.int32) for i, f in enumerate(self.files)])
        self.local_idx = np.concatenate([np.arange(f["n_windows"], dtype=np.int64) for f in self.files])

    def __len__(self):
        return len(self.labels)

    def __getstate__(self):
        # в процессы DataLoader не передаём открытые memmap: каждый откроет свои
        state = dict(self.__dict__)
        state["_mm"] = {}
        return state

    def window(self, i):
        fi = int(self.file_idx[i])
        if fi not in self._mm:
            self._mm[fi] = np.load(self.dir / self.files[fi]["windows"], mmap_mode="r")
        return np.asarray(self._mm[fi][self.local_idx[i]], dtype=np.float32)

    def channel_stats(self, file_ids=None):
        """Суммы и суммы квадратов по каналам (для глобального z-score по train)."""
        ids = range(len(self.files)) if file_ids is None else file_ids
        s = sum(np.asarray(self.files[i]["ch_sum"]) for i in ids)
        q = sum(np.asarray(self.files[i]["ch_sumsq"]) for i in ids)
        c = sum(self.files[i]["ch_count"] for i in ids)
        return s, q, c


class Normalizer:
    """global — z-score по каналам со статистиками обучающих пациентов (диссертация);
    window — z-score каждого канала внутри окна (выбрано автором в CPU-экспериментах)."""

    def __init__(self, scope="global", mean=None, std=None):
        self.scope = scope
        self.mean = None if mean is None else np.asarray(mean, np.float32)[:, None]
        self.std = None if std is None else np.asarray(std, np.float32)[:, None]

    @classmethod
    def from_caches(cls, scope, caches_and_files):
        if scope == "window":
            return cls("window")
        s = q = 0.0
        c = 0
        for cache, files in caches_and_files:
            a, b, n = cache.channel_stats(files)
            s, q, c = s + a, q + b, c + n
        mean = s / c
        std = np.sqrt(np.maximum(q / c - mean ** 2, 1e-20))
        return cls("global", mean, std)

    def __call__(self, x):
        if self.scope == "window":
            m = x.mean(axis=-1, keepdims=True)
            s = x.std(axis=-1, keepdims=True)
            s[s < 1e-8] = 1.0
            return (x - m) / s
        return (x - self.mean) / self.std

    def state(self):
        return {"scope": self.scope,
                "mean": None if self.mean is None else self.mean[:, 0].tolist(),
                "std": None if self.std is None else self.std[:, 0].tolist()}


class WindowSet(torch.utils.data.Dataset):
    """Набор окон из нескольких пациентов: items — массив пар (номер кэша, индекс окна)."""

    def __init__(self, caches, items, normalizer):
        self.caches = caches
        self.items = np.asarray(items, dtype=np.int64).reshape(-1, 2)
        self.norm = normalizer
        self.labels = np.array([caches[c].labels[i] for c, i in self.items], dtype=np.int64)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, k):
        c, i = self.items[k]
        x = self.norm(self.caches[c].window(i))
        return torch.from_numpy(np.ascontiguousarray(x, np.float32)), int(self.labels[k])


# ---------------------------------------------------------------------------
# Модель BENDR (структура SPOClab-ca/BENDR, чтобы грузились encoder.pt / contextualizer.pt)
# ---------------------------------------------------------------------------

class ChannelDropout(nn.Module):
    """Зануление канала целиком с вероятностью p (аналог Dropout2d в BENDR)."""

    def __init__(self, p):
        super().__init__()
        self.p = float(p)

    def forward(self, x):
        if not self.training or self.p == 0.0:
            return x
        keep = torch.bernoulli(x.new_full((x.shape[0], x.shape[1], 1), 1.0 - self.p))
        return x * keep


class _Identity(nn.Module):
    def forward(self, x):
        return x


class Permute(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = tuple(dims)

    def forward(self, x):
        return x.permute(*self.dims)


class ConvEncoderBENDR(nn.Module):
    """6 блоков Conv1d → ChannelDropout → GroupNorm → GELU. Чётные ядра делаются нечётными."""

    def __init__(self, in_ch, h, widths, strides, p):
        super().__init__()
        self.encoder = nn.Sequential()
        for i, (w, s) in enumerate(zip(widths, strides)):
            w = w if w % 2 else w + 1
            self.encoder.add_module(f"Encoder_{i}", nn.Sequential(
                nn.Conv1d(in_ch if i == 0 else h, h, w, stride=s, padding=w // 2),
                ChannelDropout(p), nn.GroupNorm(h // 2, h), nn.GELU()))

    def forward(self, x):
        return self.encoder(x)


class BENDRContextualizer(nn.Module):
    """Относительные позиции, проекция 512→1536, 8 слоёв трансформера (T-Fixup без LayerNorm)."""

    def __init__(self, h, d_model, ff, n_layers, n_heads, pos_k, pos_g, dropout, start_token=-5.0):
        super().__init__()
        self.start_token = start_token
        conv = nn.Conv1d(h, h, pos_k, padding=pos_k // 2, groups=pos_g)
        conv = nn.utils.weight_norm(conv, dim=2)
        self.relative_position = nn.Sequential(conv, nn.GELU())
        self.input_conditioning = nn.Sequential(
            Permute([0, 2, 1]), nn.LayerNorm(h), nn.Dropout(dropout), Permute([0, 2, 1]),
            nn.Conv1d(h, d_model, 1), Permute([2, 0, 1]))
        self.transformer_layers = nn.ModuleList()
        for _ in range(n_layers):
            layer = nn.TransformerEncoderLayer(d_model, n_heads, ff, dropout, activation="gelu")
            layer.norm1, layer.norm2 = _Identity(), _Identity()
            self.transformer_layers.append(layer)
        self.norm = nn.LayerNorm(d_model)
        self.mask_replacement = nn.Parameter(torch.zeros(h))
        self.output_layer = nn.Conv1d(d_model, h, 1)

    def forward(self, x):
        x = x + self.relative_position(x)
        x = self.input_conditioning(x)                       # [T, B, 1536]
        tok = x.new_full((1, x.shape[1], x.shape[2]), float(self.start_token))
        x = torch.cat([tok, x], 0)
        for layer in self.transformer_layers:
            x = layer(x)
        return self.output_layer(x.permute(1, 2, 0))         # [B, 512, T+1]


class BENDRClassifier(nn.Module):
    """Энкодер + контекстуализатор + линейная голова на два класса (softmax)."""

    def __init__(self, cfg):
        super().__init__()
        m = cfg["model"]
        h = int(m["encoder_h"])
        self.feature = str(m.get("feature", "ctx_first"))
        self.encoder = ConvEncoderBENDR(int(m["n_channels"]), h, m["enc_width"], m["enc_downsample"],
                                        float(m["channel_dropout_p"]))
        self.contextualizer = BENDRContextualizer(
            h, int(m["d_model"]), int(m["ff_dim"]), int(m["n_layers"]), int(m["n_heads"]),
            int(m["relative_pos_kernel"]), int(m["relative_pos_groups"]), float(m["contextualizer_dropout"]))
        self.head = nn.Sequential(nn.Linear(h, 2), nn.Softmax(dim=-1))

    def features(self, x):
        z = self.encoder(x)
        if self.feature == "enc_mean":
            return z.mean(-1)
        c = self.contextualizer(z)
        return c[:, :, 0] if self.feature == "ctx_first" else c[:, :, 1:].mean(-1)

    def forward(self, x):
        return self.head(self.features(x))

    def load_pretrained(self, weights_dir):
        """Подставляет официальные веса BENDR; голова остаётся случайной."""
        wd = resolve_path(weights_dir)
        rep = {}
        for name, module in (("encoder", self.encoder), ("contextualizer", self.contextualizer)):
            path = wd / f"{name}.pt"
            if not path.is_file():
                raise FileNotFoundError(f"Нет весов {path}. Запустите prepare_data.py --weights.")
            state = torch.load(path, map_location="cpu", weights_only=True)
            missing, unexpected = module.load_state_dict(state, strict=False)
            rep[name] = {"missing": list(missing), "unexpected": list(unexpected)}
        return rep


def configure_freezing(model, mode, unfreeze_last=0):
    """Режимы: full — всё обучается (диссертация); encoder — заморожен свёрточный энкодер;
    backbone — заморожены энкодер и контекстуализатор, учится только голова;
    partial — заморожен энкодер и первые слои трансформера, учатся последние
    unfreeze_last слоёв, выходная проекция и голова (выбрано автором)."""
    for p in model.parameters():
        p.requires_grad = True
    frozen = []
    if mode in ("encoder", "backbone", "partial"):
        frozen.append(model.encoder)
    if mode == "backbone":
        frozen.append(model.contextualizer)
    if mode == "partial":
        ctx = model.contextualizer
        n = len(ctx.transformer_layers)
        k = max(0, min(int(unfreeze_last), n))
        frozen += [ctx.relative_position, ctx.input_conditioning]
        frozen += list(ctx.transformer_layers[: n - k])
        ctx.mask_replacement.requires_grad = False
    if mode not in ("full", "encoder", "backbone", "partial"):
        raise ValueError(f"Неизвестный режим заморозки: {mode}")
    for mod in frozen:
        for p in mod.parameters():
            p.requires_grad = False
    model._frozen_modules = frozen
    n_all = sum(p.numel() for p in model.parameters())
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return n_tr, n_all


def train_mode(model, flag=True):
    """model.train(), но замороженные части остаются в eval (без dropout)."""
    model.train(flag)
    for mod in getattr(model, "_frozen_modules", []):
        mod.eval()


# ---------------------------------------------------------------------------
# Функция потерь SSWCE
# ---------------------------------------------------------------------------

class SSWCE(nn.Module):
    """L = CE + α(1 − специфичность) + β(1 − чувствительность), α = 0,3, β = 0,5.

    Метрики считаются «мягко» по вероятностям, чтобы проходил градиент.
    class_weights — веса классов в CE (режим balancing=class_weights).
    """

    def __init__(self, alpha=0.3, beta=0.5, class_weights=None):
        super().__init__()
        self.alpha, self.beta = float(alpha), float(beta)
        w = None if class_weights is None else torch.as_tensor(class_weights, dtype=torch.float32)
        self.register_buffer("w", w if w is not None else torch.ones(2), persistent=False)
        self.use_w = class_weights is not None

    def forward(self, probs, target):
        probs = probs.float()
        logp = torch.log(probs.clamp(min=1e-7))
        ce = F.nll_loss(logp, target, weight=self.w if self.use_w else None)
        p1, y = probs[:, 1], target.float()
        sens = (p1 * y).sum() / y.sum().clamp(min=1.0)
        spec = ((1 - p1) * (1 - y)).sum() / (1 - y).sum().clamp(min=1.0)
        return ce + self.alpha * (1 - spec) + self.beta * (1 - sens)


# ---------------------------------------------------------------------------
# Пост-обработка и метрики
# ---------------------------------------------------------------------------

def majority_voting(pred, width=3):
    """Голосование большинством в скользящем окне нечётной ширины (края — повтором)."""
    y = np.asarray(pred, np.int64)
    if width < 1 or width % 2 == 0:
        raise ValueError(f"Окно голосования должно быть нечётным, получено {width}.")
    if width == 1 or len(y) == 0:
        return y.copy()
    half = width // 2
    pad = np.pad(y, (half, half), mode="edge")
    sums = np.convolve(pad, np.ones(width, np.int64), mode="valid")
    return (sums > half).astype(np.int64)


def count_fp_groups(pred, true):
    """Подряд идущие ложные срабатывания считаются одним событием."""
    fp = ((np.asarray(pred) == 1) & (np.asarray(true) == 0)).astype(np.int8)
    return int(((fp[1:] == 1) & (fp[:-1] == 0)).sum() + (fp[:1] == 1).sum()) if len(fp) else 0


def youden_threshold(y, p1):
    """Порог по валидации: максимум J = чувствительность + специфичность − 1 (выбрано автором).

    Перебираются все различные значения вероятности (roc_curve), а не сетка,
    поэтому порог находится и при очень узком разбросе вероятностей.
    """
    from sklearn.metrics import roc_curve
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return 0.5, None
    fpr, tpr, thr = roc_curve(y, p1)
    j = tpr - fpr
    k = int(np.argmax(j[1:]) + 1)          # thr[0] = +inf, его пропускаем
    return float(thr[k]), float(j[k])


def safe_auc(y, p1):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, p1))


def detection_metrics(y, p1, file_ids, starts, events, threshold, vote_width, window_sec):
    """Метрики одного пациента. Голосование и группировка FP — внутри каждого файла.

    events — словарь {номер файла: [(начало, конец), ...]} для событийной чувствительности.
    Пропуски окон постиктала внутри файла при голосовании не учитываются (как в НИР-3).
    """
    y = np.asarray(y, np.int64)
    pred = (np.asarray(p1) >= threshold).astype(np.int64)
    out_pred = np.empty_like(pred)
    n_fp_groups, ev_hit, ev_total = 0, 0, 0
    for f in np.unique(file_ids):
        m = file_ids == f
        pf = majority_voting(pred[m], vote_width)
        out_pred[m] = pf
        n_fp_groups += count_fp_groups(pf, y[m])
        st = starts[m]
        for a, b in events.get(int(f), []):
            ov = (st < b) & (st + window_sec > a)
            if ov.any():
                ev_total += 1
                ev_hit += int((pf[ov] == 1).any())
    tp = int(((out_pred == 1) & (y == 1)).sum())
    tn = int(((out_pred == 0) & (y == 0)).sum())
    fp = int(((out_pred == 1) & (y == 0)).sum())
    fn = int(((out_pred == 0) & (y == 1)).sum())
    sens = tp / (tp + fn) if tp + fn else None
    spec = tn / (tn + fp) if tn + fp else None
    hours = (tn + fp) * window_sec / 3600.0
    return {
        "vote_width": int(vote_width), "threshold": float(threshold),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "sensitivity": sens, "specificity": spec,
        # FP/ч по формуле диссертации: (1 − специфичность) · 3600 / длительность окна
        "fp_per_hour": None if spec is None else (1 - spec) * 3600.0 / window_sec,
        "grouped_fp_per_hour": n_fp_groups / hours if hours > 0 else None,
        "event_sensitivity": ev_hit / ev_total if ev_total else None,
        "n_events": ev_total, "interictal_hours": hours,
    }
