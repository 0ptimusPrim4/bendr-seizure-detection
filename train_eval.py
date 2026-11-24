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
Вход модели — как при предобучении BENDR (DN3 MappingDeep1010): окно min-max в [-1, 1] + канал SCALE (--norm bendr).
Голова: BatchNorm (пересчёт статистики перед оценкой) → Linear, fp32, + штраф нормы признаков --feat-reg; --feature auto: при backbone —
линейный зонд замороженного BENDR на первом фолде, при дообучении — ctx_mean. dropout энкодера/трансформера 0, как в дообучении BENDR.
AdamW с раздельным lr (--lr-head / --lr-backbone), warmup + косинус до --min-lr-ratio, clip 1.0, лучшая модель по val AUC.
Проверка, что модель вообще учится (и таблица AUC по всем признакам): python train_eval.py --sanity-overfit
Метрики до/после пост-обработки (голосование в нечётном окне, группировка FP) — <out>/<run>/summary.csv.
"""

import argparse
import hashlib
import importlib.util
import json
import math
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
# Нормализация как в BENDR/DN3 (MappingDeep1010): каждое окно min-max в [-1, 1] по всем 19 каналам ЭЭГ,
# 20-й канал SCALE = 2·(min(размах окна / MAX_SCALE, 1) − 0,5). В DN3 MAX_SCALE = data_max − data_min датасета;
# для CHB-MIT (данные в вольтах) берём 2 мВ, как ±1 мВ в конфиге MMIDB BENDR — выбрано автором, флаг --max-scale.
MAX_SCALE = 2e-3
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
    def __init__(self, c_in=20, h=512, widths=(3, 2, 2, 2, 2, 2), strides=(3, 2, 2, 2, 2, 2), p=0.0):
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
        self.mask_replacement = nn.Parameter(torch.zeros(h))   # используется только при предобучении
        self.output_layer = nn.Conv1d(d, h, 1)

    def forward(self, x):
        x = self.input_conditioning(x + self.relative_position(x))
        x = torch.cat([x.new_full((1, x.shape[1], x.shape[2]), -5.0), x], 0)  # стартовый токен
        for layer in self.transformer_layers:
            x = layer(x)
        return self.output_layer(x.permute(1, 2, 0))


# Размерность признаков для головы
FEATURES = {"ctx_last": 512, "ctx_mean": 512, "ctx_meanmax": 1024, "ctx_first": 512,
            "enc_mean": 512, "enc_meanmax": 1024, "enc_pool4": 2048}


def pool_features(z, c=None):
    """Все варианты признаков из выхода энкодера z [B,512,T] и контекстуализатора c [B,512,T+1]."""
    out = {"enc_mean": z.mean(-1), "enc_meanmax": torch.cat([z.mean(-1), z.amax(-1)], 1),
           "enc_pool4": F.adaptive_avg_pool1d(z, 4).flatten(1)}   # enc_pool4 — как LinearHeadBENDR
    if c is not None:
        t = c[:, :, 1:]
        out.update({"ctx_first": c[:, :, 0], "ctx_last": c[:, :, -1],   # ctx_last — как BENDRClassification
                    "ctx_mean": t.mean(-1), "ctx_meanmax": torch.cat([t.mean(-1), t.amax(-1)], 1)})
    return out


def make_head(dim, kind="linear", dropout=0.0, norm="bn"):
    """Голова на логитах. norm: bn — BatchNorm1d по каждому признаку: убирает общую для всех окон составляющую
    признаков BENDR (различия между окнами ~10 % от неё), голова учится быстро. При дообучении признаки дрейфуют
    и накопленные средние BatchNorm отстают → в eval-режиме выход насыщается (p = 0/1, val loss 100+). Поэтому
    перед каждой оценкой статистика пересчитывается на текущей магистрали (recalibrate_bn).
    ln — LayerNorm по окну (без статистики), но на признаках BENDR учится медленно и неустойчиво при большом lr.
    fixed — та же стандартизация, но среднее/σ считаются один раз по предобученной модели и не меняются.
    Норма признаков ограничена, но σ у признаков крошечные, и малый сдвиг общей составляющей превращается
    в большие скачки смещения логитов (p̄ 0,02 ↔ 0,98 между эпохами) — поэтому по умолчанию bn + --feat-reg:
    bn не видит масштаба признаков, и рост нормы сдерживает отдельный штраф в fit."""
    nl = nn.BatchNorm1d(dim) if norm in ("bn", "fixed") else nn.LayerNorm(dim)
    if kind == "mlp":
        head = nn.Sequential(nl, nn.Linear(dim, 256), nn.GELU(), nn.Dropout(dropout), nn.Linear(256, 2))
    else:
        head = nn.Sequential(nl, nn.Dropout(dropout), nn.Linear(dim, 2))
    # последний слой с нуля: старт с p = 0,5 (на стандартизованных признаках xavier даёт логиты ~10 и loss ~7)
    nn.init.zeros_(head[-1].weight)
    nn.init.zeros_(head[-1].bias)
    return head


class BENDR(nn.Module):
    def __init__(self, feature="ctx_last", head="linear", head_dropout=0.0, dropout=0.0, head_norm="bn"):
        super().__init__()
        self.feature = feature
        # dropout энкодера и трансформера: в дообучении BENDR (BENDRClassification) он равен 0
        self.encoder, self.contextualizer = Encoder(p=dropout), Contextualizer(dropout=dropout)
        self.head = make_head(FEATURES[feature], head, head_dropout, head_norm)
        self.head_fixed = head_norm == "fixed"

    ctx_fp32 = False   # --amp-scope encoder: трансформер считается в fp32 даже при AMP

    def _ctx(self, z):
        if self.ctx_fp32:
            with torch.autocast(z.device.type, enabled=False):
                return self.contextualizer(z.float())
        return self.contextualizer(z)

    def all_features(self, x):
        z = self.encoder(x)
        return pool_features(z, self._ctx(z))

    def features(self, x):
        z = self.encoder(x)
        if self.feature.startswith("enc"):
            return pool_features(z)[self.feature]
        # ctx_first — выход стартового токена; с весами v0.1-alpha почти не зависит от входа (std ≈ 0).
        return pool_features(z, self._ctx(z))[self.feature]

    def load_pretrained(self, enc_path, ctx_path, log=print):
        """Загрузка весов с отчётом; если не совпало больше половины ключей — остановка."""
        for name, mod, path in (("encoder", self.encoder, enc_path), ("contextualizer", self.contextualizer, ctx_path)):
            state = torch.load(path, map_location="cpu", weights_only=True)
            own = mod.state_dict()
            shape_bad = [k for k in state if k in own and own[k].shape != state[k].shape]
            for k in shape_bad:
                state.pop(k)
            miss, unexp = mod.load_state_dict(state, strict=False)
            loaded = len(own) - len(miss)
            log(f"веса {name}: загружено {loaded}/{len(own)} тензоров; нет в файле {len(miss)}, "
                f"лишних {len(unexp)}, не совпала форма {len(shape_bad)}")
            if miss or unexp or shape_bad:
                log(f"  нет: {list(miss)[:5]} лишние: {list(unexp)[:5]} форма: {shape_bad[:5]}")
            if loaded < 0.5 * len(own):
                sys.exit(f"ОШИБКА: {path} почти не подошёл к модели ({loaded}/{len(own)}) — "
                         f"энкодер остался бы случайным. Проверьте, что это веса BENDR v0.1-alpha.")


def set_freeze(model, mode, k):
    """Что учится в каждом режиме.
    Признак ctx_*: full — всё; encoder — контекстуализатор (энкодер заморожен); partial — последние k слоёв
    трансформера + выходной слой; backbone — только голова.
    Признак enc_* (контекстуализатор не нужен и всегда заморожен): full — весь энкодер; partial — последние k
    свёрточных блоков энкодера (из 6); encoder и backbone — только голова."""
    for p in model.parameters():
        p.requires_grad = True
    ctx, enc = model.contextualizer, model.encoder
    blocks = list(enc.encoder)
    if model.feature.startswith("enc"):
        frozen = {"full": [ctx], "encoder": [enc, ctx], "backbone": [enc, ctx],
                  "partial": [ctx, *blocks[:max(0, len(blocks) - k)]]}[mode]
    else:
        frozen = {"full": [], "encoder": [enc], "backbone": [enc, ctx],
                  "partial": [enc, ctx.relative_position, ctx.input_conditioning,
                              *ctx.transformer_layers[:max(0, len(ctx.transformer_layers) - k)]]}[mode]
    for m in frozen:
        for p in m.parameters():
            p.requires_grad = False
    ctx.mask_replacement.requires_grad = False   # в дообучении не участвует
    model._frozen = frozen
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_mode(model, flag):
    model.train(flag)
    for m in model._frozen:   # замороженные части без dropout
        m.eval()
    if getattr(model, "head_fixed", False):   # фиксированная стандартизация: статистика не обновляется
        for m in model.head.modules():
            if isinstance(m, nn.BatchNorm1d):
                m.eval()


class Criterion(nn.Module):
    """sswce: CE + α(1 − спец.) + β(1 − чувств.), α = 0,3, β = 0,5 (Benfenati, ур. 5.9);
    ce — обычная кросс-энтропия; focal — фокальная (γ = --focal-gamma, выбрано автором).
    weights — веса классов (только при --balancing class_weights, чтобы не балансировать дважды)."""

    def __init__(self, kind="sswce", weights=None, alpha=0.3, beta=0.5, gamma=2.0):
        super().__init__()
        self.kind, self.a, self.b, self.g = kind, alpha, beta, gamma
        self.w = None if weights is None else torch.tensor(weights, dtype=torch.float32)

    def forward(self, logits, y):
        logits = logits.float()
        w = None if self.w is None else self.w.to(logits.device)
        logp = F.log_softmax(logits, -1)
        if self.kind == "focal":
            lp = logp.gather(1, y[:, None])[:, 0]
            loss = -((1 - lp.exp()) ** self.g) * lp
            if w is not None:
                loss = loss * w[y]
            return loss.mean()
        ce = F.nll_loss(logp, y, weight=w)
        if self.kind == "ce":
            return ce
        p1, yf = logp[:, 1].exp(), y.float()
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
        kind = self.norm[0]
        if kind == "bendr":     # как MappingDeep1010 в DN3/BENDR
            e = x[:19]
            lo, hi = float(e.min()), float(e.max())
            r = hi - lo
            x = np.empty_like(x)
            x[:19] = 2 * (e - lo) / r - 1 if r > 0 else 0
            x[19] = 2 * (min(r / self.norm[1], 1.0) - 0.5)
        elif kind == "window":  # z-score каждого канала внутри окна
            s = x.std(-1, keepdims=True)
            x = (x - x.mean(-1, keepdims=True)) / np.where(s < 1e-8, 1, s)
        else:                   # глобальный z-score по обучающим пациентам
            x = (x - self.norm[1]) / self.norm[2]
        return torch.from_numpy(x.astype(np.float32)), int(self.labels[k])


def global_norm(parts):
    s = q = 0.0
    n = 0
    for cache, files in parts:
        a, b, c = cache.channel_stats(files)
        s, q, n = s + a, q + b, n + c
    m = s / n
    return (m[:, None].astype(np.float32), np.sqrt(np.maximum(q / n - m ** 2, 1e-20))[:, None].astype(np.float32))


def make_norm(a, fold):
    if a.norm == "bendr":
        return ("bendr", a.max_scale)
    if a.norm == "window":
        return ("window",)
    return ("global", *global_norm(fold["norm"]))


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

def amp_dtype(a, device):
    """bf16, если GPU умеет (у bf16 нет переполнений fp16), иначе fp16 + GradScaler; на CPU — выкл."""
    if a.amp == "off" or (device.type != "cuda" and a.amp != "bf16"):
        return None   # на CPU — fp32 (bf16 только если явно указан, для проверки)
    if a.amp == "auto":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return {"bf16": torch.bfloat16, "fp16": torch.float16}[a.amp]


def loader(ds, y, a, train, device, workers):
    kw = dict(batch_size=a.batch_size, num_workers=workers, pin_memory=device.type == "cuda")
    if train and a.balancing == "sampler":   # WeightedRandomSampler, веса обратны частоте класса
        cnt = np.maximum(np.bincount(y, minlength=2), 1)
        s = WeightedRandomSampler(torch.as_tensor(1.0 / cnt[y]), a.samples_per_epoch or len(y), replacement=True)
        return DataLoader(ds, sampler=s, **kw)
    return DataLoader(ds, shuffle=train, **kw)


def logits_of(model, x, dt, from_feats, with_feats=False):
    """Магистраль BENDR — в смешанной точности, голова и softmax — всегда в fp32."""
    if from_feats:
        f = x
    else:
        with torch.autocast(x.device.type, dtype=dt or torch.float32, enabled=dt is not None):
            f = model.features(x)
    f = f.float()
    lg = model.head(f)
    return (lg, f) if with_feats else lg


@torch.no_grad()
def predict(model, dl, crit, device, dt, from_feats):
    """Возвращает (средний loss, p(приступ)) — один проход в eval-режиме."""
    train_mode(model, False)
    ps, tot, n = [], 0.0, 0
    for x, y in dl:
        x, y = x.to(device, non_blocking=True), y.to(device).long()
        lg = logits_of(model, x, dt, from_feats)
        tot, n = tot + float(crit(lg, y)) * len(y), n + len(y)
        ps.append(torch.softmax(lg, -1)[:, 1].cpu().numpy())
    return tot / max(n, 1), (np.concatenate(ps) if ps else np.empty(0))


@torch.no_grad()
def extract(model, dl, device, dt):
    train_mode(model, False)
    out = []
    for x, _ in dl:
        with torch.autocast(device.type, dtype=dt or torch.float32, enabled=dt is not None):
            out.append(model.features(x.to(device, non_blocking=True)).float().cpu())
    return torch.cat(out)


def youden(y, p):
    from sklearn.metrics import roc_curve
    if len(np.unique(y)) < 2 or not np.isfinite(p).all():
        return 0.5
    fpr, tpr, thr = roc_curve(y, p)
    return float(thr[1:][np.argmax((tpr - fpr)[1:])])


def describe(y, p):
    """AUC, порог Юдена и чувствительность/специфичность при нём, статистика вероятностей."""
    finite = bool(np.isfinite(p).all())
    thr = youden(y, p) if finite else 0.5
    pr = p >= thr
    sens = float((pr & (y == 1)).sum() / max(1, (y == 1).sum()))
    spec = float((~pr & (y == 0)).sum() / max(1, (y == 0).sum()))
    return {"auc": auc(y, p) if finite else None, "thr": thr, "sens": sens, "spec": spec,
            "p_mean": float(np.nanmean(p)), "p_std": float(np.nanstd(p)), "finite": finite}


def fit(model, tl, vl, yv, crit, a, device, dt, ckpt, log, from_feats=False, epochs=None, patience=None):
    """Цикл обучения: AdamW с разными lr для головы и магистрали, warmup + косинус до lr·--min-lr-ratio,
    клиппинг градиента, защита от NaN, выбор лучшей эпохи по val AUC."""
    epochs, patience = epochs or a.epochs, patience or a.patience
    head = list(model.head.parameters())
    head_ids = {id(p) for p in head}
    bb = [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids]
    groups = [{"params": head, "lr": a.lr_head, "weight_decay": a.weight_decay}]
    if bb and not from_feats:
        # weight decay магистрали только на матрицах весов (не на bias/нормировках) — сдерживает рост нормы
        groups.append({"params": [p for p in bb if p.ndim >= 2], "lr": a.lr_backbone, "weight_decay": a.wd_backbone})
        groups.append({"params": [p for p in bb if p.ndim < 2], "lr": a.lr_backbone, "weight_decay": 0.0})
    groups = [g for g in groups if g["params"]]
    opt = torch.optim.AdamW(groups)
    spe = max(1, len(tl))
    total, warm = epochs * spe, int(a.warmup * spe)
    gradual = a.unfreeze_epoch > 0 and bb and not from_feats
    start_bb = a.unfreeze_epoch * spe if gradual else 0

    def lr_curve(start):
        # 0 до start; линейный разогрев --warmup эпох от start; косинус до lr·--min-lr-ratio к концу обучения
        def f(s):
            s -= start
            if s < 0:
                return 0.0
            if s < warm:
                return (s + 1) / warm
            return a.min_lr_ratio + (1 - a.min_lr_ratio) * 0.5 * (
                1 + math.cos(math.pi * min(1.0, (s - warm) / max(1, total - start - warm))))
        return f
    # у магистрали свой разогрев с момента разморозки (иначе после разморозки сразу полный lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, [lr_curve(0)] + [lr_curve(start_bb)] * (len(groups) - 1))
    probe_x = next(iter(vl))[0][:64] if a.diag else None
    scaler = torch.amp.GradScaler("cuda", enabled=dt == torch.float16)
    params = [p for g in groups for p in g["params"]]
    names = {id(p): n for n, p in model.named_parameters()}
    save_names = {names[id(p)] for p in params}
    # постепенная разморозка: первые --unfreeze-epoch эпох учится только голова
    if gradual:
        for p in bb:
            p.requires_grad = False
        log(f"  эпохи 1–{a.unfreeze_epoch}: учится только голова, затем размораживается магистраль")
    if getattr(model, "head_fixed", False):
        recalibrate_bn(model, tl, device, dt, from_feats, max(a.bn_recal, 1))
        log(f"  голова: стандартизация признаков зафиксирована по предобученной модели ({max(a.bn_recal, 1)} батчей)")
    reg_on = a.feat_reg > 0 and bool(bb) and not from_feats
    if reg_on:
        with torch.no_grad():
            train_mode(model, False)
            xb = next(iter(tl))[0].to(device)
            _, f0 = logits_of(model, xb, dt, from_feats, with_feats=True)
            ref_norm = float(f0.norm(dim=1).mean())
        log(f"  штраф нормы признаков: λ {a.feat_reg:g}, опорная |признак| {ref_norm:.3g} (предобученная модель)")
    best, bad, hist, bad_steps = -float("inf"), 0, [], 0
    norm0 = None
    for ep in range(1, epochs + 1):
        if gradual and ep == a.unfreeze_epoch + 1:
            for p in bb:
                p.requires_grad = True
            log(f"  эпоха {ep}: магистраль разморожена, lr магистрали разогревается {a.warmup:g} эп. до {a.lr_backbone:g}")
        t0 = time.time()
        train_mode(model, True)
        tot = n = 0
        for x, y in tl:
            if len(y) < 2:              # BatchNorm в голове не работает на батче из одного окна
                continue
            x, y = x.to(device, non_blocking=True), y.to(device).long()
            lg, f = logits_of(model, x, dt, from_feats, with_feats=True)
            loss = crit(lg, y)
            if reg_on and any(p.requires_grad for p in bb):
                # штраф за уход нормы признаков от предобученной: BatchNorm-голова не видит масштаба,
                # и без штрафа магистраль раздувает признаки (|f| 0,14 → 9e4 на chb01)
                loss = loss + a.feat_reg * torch.log(f.norm(dim=1).mean() / ref_norm) ** 2
            opt.zero_grad(set_to_none=True)
            if not torch.isfinite(loss):
                bad_steps += 1          # шаг с NaN/inf пропускается
                sched.step()
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_([p for p in params if p.grad is not None], a.clip)
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot, n = tot + float(loss.detach()) * len(y), n + len(y)
        tr = tot / max(n, 1)
        if not getattr(model, "head_fixed", False):
            recalibrate_bn(model, tl, device, dt, from_feats, a.bn_recal)
        vloss, pv = predict(model, vl, crit, device, dt, from_feats)
        d = describe(yv, pv)
        hist.append({"epoch": ep, "train_loss": tr, "val_loss": vloss, **d, "nan_steps": bad_steps})
        score = (d["auc"] if d["auc"] is not None else -1) if a.monitor == "auc" else -vloss
        star = ""
        if score > best + 1e-4:
            best, bad, star = score, 0, " *"
            torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()
                        if k in save_names or k.startswith("head.")}, ckpt)   # + статистика BatchNorm головы
        else:
            bad += 1
        auc_s = "NaN" if d["auc"] is None else f"{d['auc']:.3f}"
        lrs = "/".join(f"{g['lr']:.1e}" for g in opt.param_groups)
        log(f"  эпоха {ep:3d}: train {tr:.4f} | val {vloss:.4f} AUC {auc_s} "
            f"Se {d['sens']:.2f} Sp {d['spec']:.2f} (порог {d['thr']:.3f}) | p {d['p_mean']:.3f}±{d['p_std']:.4f} "
            f"| lr {lrs} | {time.time() - t0:.0f} с{star}")
        if probe_x is not None:
            line, fn = diagnose(model, probe_x.to(device), dt, from_feats)
            log("  " + line)
            if norm0 is None and (not gradual or ep > a.unfreeze_epoch):
                norm0 = fn   # норма признаков в первую эпоху обучения магистрали
            elif norm0 and fn > 100 * norm0:
                log(f"  ВНИМАНИЕ: норма признаков выросла в {fn / norm0:.0f} раз ({norm0:.3g} → {fn:.3g}) — магистраль "
                    f"«раздувает» признаки вместо обучения. Уменьшите --lr-backbone, увеличьте --wd-backbone, "
                    f"проверьте --head-norm fixed; при bf16 — --amp-scope encoder.")
        if not d["finite"]:
            log("  ВНИМАНИЕ: в предсказаниях NaN/inf — попробуйте --amp off или меньше --lr-backbone")
        elif d["p_std"] < 1e-4:
            log("  ВНИМАНИЕ: предсказания на валидации почти постоянны — модель схлопнулась в константу "
                "(проверьте --feature, --lr-backbone, --sanity-overfit)")
        if bad_steps:
            log(f"  пропущено шагов с NaN/inf: {bad_steps}")
        if bad >= patience:
            log("  ранняя остановка")
            break
    if ckpt.is_file():
        model.load_state_dict(torch.load(ckpt, weights_only=True), strict=False)
    return hist


@torch.no_grad()
def recalibrate_bn(model, dl, device, dt, from_feats, max_batches):
    """Пересчёт статистики BatchNorm головы на текущей магистрали (точное среднее по max_batches батчам train).
    Без этого eval-режим использует устаревшие средние, накопленные до сдвига признаков."""
    bns = [m for m in model.head.modules() if isinstance(m, nn.BatchNorm1d)]
    if not bns or max_batches <= 0:
        return
    fixed = getattr(model, "head_fixed", False)
    model.head_fixed = False
    was = model.training
    train_mode(model, False)
    for m in bns:
        m.reset_running_stats()
        m.momentum = None          # кумулятивное среднее
        m.train()
    for i, (x, _) in enumerate(dl):
        if i >= max_batches:
            break
        if len(x) > 1:
            logits_of(model, x.to(device, non_blocking=True), dt, from_feats)
    for m in bns:
        m.momentum = 0.1
    model.head_fixed = fixed
    train_mode(model, was)


@torch.no_grad()
def diagnose(model, x, dt, from_feats):
    """Одна строка диагностики на фиксированном val-батче: величина признаков, расхождение bf16/fp16 с fp32
    и расхождение логитов головы в train- и eval-режиме (BatchNorm, dropout)."""
    import copy
    was = model.training
    train_mode(model, False)
    if from_feats:
        f = x.float()
        prec = ""
    else:
        with torch.autocast(x.device.type, dtype=dt or torch.float32, enabled=dt is not None):
            f = model.features(x).float()
        prec = ""
        if dt is not None:
            f32 = model.features(x).float()
            prec = f" | {str(dt).split('.')[-1]}/fp32 Δ {float((f - f32).norm() / (f32.norm() + 1e-12)):.3f}"
            train_mode(model, True)
            with torch.autocast(x.device.type, dtype=dt, enabled=True):
                ft = model.features(x).float()
            train_mode(model, False)
        else:
            train_mode(model, True)
            ft = model.features(x).float()
            train_mode(model, False)
        prec += f" | признак train/eval Δ {float((ft - f).norm() / (f.norm() + 1e-12)):.3f}"
    le = model.head(f)
    h = copy.deepcopy(model.head).train()   # копия: running stats настоящей головы не трогаем
    if getattr(model, "head_fixed", False):
        h.eval()
    lt = h(f)
    train_mode(model, was)
    spread = float(f.std(0).mean() / (f.abs().mean() + 1e-12))
    fn = float(f.norm(dim=1).mean())
    return (f"диагн.: |признак| {fn:.3g}, max {float(f.abs().max()):.3g}, разброс {spread:.3f}"
            f"{prec} | логит eval {float(le.abs().max()):.3g}, train/eval Δлогит {float((lt - le).abs().max()):.3g}"), fn


def build_model(a, device, log, verbose):
    model = BENDR(a.feature, a.head, a.head_dropout, a.dropout, a.head_norm)
    model.load_pretrained(a.encoder, a.contextualizer, log if verbose else (lambda *_: None))
    model.ctx_fp32 = a.amp_scope == "encoder"
    n_tr = set_freeze(model, a.freeze, a.unfreeze_last)
    n_head = sum(p.numel() for p in model.head.parameters())
    tr_enc = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
    tr_ctx = sum(p.numel() for p in model.contextualizer.parameters() if p.requires_grad)
    if verbose:
        log(f"обучаемые параметры: голова {n_head:,}, энкодер {tr_enc:,}, контекстуализатор {tr_ctx:,} "
            f"(freeze {a.freeze}, признак {a.feature}, нормировка головы {a.head_norm})")
    if a.freeze != "backbone" and n_tr <= n_head:
        log(f"ВНИМАНИЕ: freeze {a.freeze} с признаком {a.feature} учит ТОЛЬКО голову ({n_head:,} параметров) — "
            f"это то же, что --freeze backbone. Для дообучения возьмите признак ctx_* или --freeze partial/full.")
    return model.to(device), n_tr


def make_criterion(a, ytr):
    cw = None
    if a.balancing == "class_weights":
        cw = (len(ytr) / (2 * np.maximum(np.bincount(ytr, minlength=2), 1))).tolist()
    return Criterion(a.loss, cw, gamma=a.focal_gamma)


def input_stats(dl, log, tag):
    x, y = next(iter(dl))
    x = x.float()
    log(f"{tag}: вход батча {tuple(x.shape)} mean {x.mean():.3f} std {x.std():.3f} min {x.min():.2f} "
        f"max {x.max():.2f}, приступов в батче {int(y.sum())}/{len(y)}, конечные: {bool(torch.isfinite(x).all())}")


def feature_check(f, log, tag):
    rel = float(f.std(0).mean() / (f.abs().mean() + 1e-12))
    log(f"{tag}: признаки {tuple(f.shape)}, разброс между окнами {rel:.4f} (относительный)")
    if rel < 1e-3:
        log(f"  ВНИМАНИЕ: признак {tag} почти одинаков для всех окон — голова не сможет ничему научиться. "
            f"Возьмите --feature ctx_mean / ctx_last / enc_mean.")


def train_fold(fold, frac, caches, a, device, run_dir, log, first=False):
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
    if a.val_max and len(fold["val"]) > a.val_max and "val_full" not in fold:
        # для ежепоэпохной оценки: все окна с приступом + случайный фон, всего --val-max окон
        yv0 = lab(fold["val"])
        pos, neg = np.where(yv0 == 1)[0], np.where(yv0 == 0)[0]
        keep = np.sort(np.concatenate([pos, rng.choice(neg, max(0, a.val_max - len(pos)), replace=False)]))
        fold["val_full"], fold["val"] = fold["val"], fold["val"][keep]
        log(f"{tag}: val урезан до {len(keep)} окон (все {len(pos)} с приступом), --val-max {a.val_max}")
    ytr, yva, yte = lab(tr_it), lab(fold["val"]), lab(fold["test"])
    log(f"{tag}: train {len(ytr)} (приступ {ytr.sum()}, {ytr.mean():.2%}), val {len(yva)} ({yva.sum()}), "
        f"test {len(yte)} ({yte.sum()}); балансировка {a.balancing}, loss {a.loss}")
    if not ytr.sum() or not yva.sum():
        return log(f"{tag}: нет приступов в train/val — пропуск")
    norm = make_norm(a, fold)
    ds = lambda it: Windows(cl, it, norm)
    crit = make_criterion(a, ytr)
    model, n_tr = build_model(a, device, log, first)
    dt = amp_dtype(a, device)
    log(f"{tag}: freeze {a.freeze}, обучаемых {n_tr:,}, признак {a.feature}, AMP {dt}, "
        f"lr голова {a.lr_head:g} / магистраль {a.lr_backbone:g}")
    ckpt = out / "best.pt"
    if a.freeze == "backbone":   # признаки BENDR один раз на фолд, дальше учится только голова
        if "feats" not in fold:
            t0 = time.time()
            fold["feats"] = [extract(model, DataLoader(ds(fold[k]), a.batch_size, num_workers=a.workers), device, dt)
                             for k in ("train", "val", "test")]
            log(f"{fold['name']}: признаки за {time.time() - t0:.0f} с")
            feature_check(fold["feats"][0], log, a.feature)
        ftr, fva, fte = fold["feats"]
        if sel is not None:
            ftr = ftr[torch.as_tensor(sel)]
        tl = loader(TensorDataset(ftr, torch.as_tensor(ytr)), ytr, a, True, device, 0)
        vl = DataLoader(TensorDataset(fva, torch.as_tensor(yva)), a.batch_size)
        tel = DataLoader(TensorDataset(fte, torch.as_tensor(yte)), a.batch_size)
        hist = fit(model, tl, vl, yva, crit, a, device, dt, ckpt, log, from_feats=True)
        fe = True
    else:
        tl = loader(ds(tr_it), ytr, a, True, device, a.workers)
        vl, tel = (loader(ds(fold[k]), None, a, False, device, a.workers) for k in ("val", "test"))
        if first:
            input_stats(tl, log, tag)
        hist = fit(model, tl, vl, yva, crit, a, device, dt, ckpt, log)
        fe = False
    _, pva = predict(model, vl, crit, device, dt, fe)
    _, pte = predict(model, tel, crit, device, dt, fe)
    thr = youden(yva, pva) if a.threshold == "youden" else 0.5
    tc, ti = cl[int(fold["test"][0, 0])], fold["test"][:, 1]
    np.savez_compressed(out / "preds.npz", p=pte, y=yte, file_idx=tc.file_idx[ti], starts=tc.starts[ti])
    events = {int(i): tc.files[i]["seizures"] for i in np.unique(tc.file_idx[ti])}
    info = {"fold": fold["name"], "fraction": frac, "freeze": a.freeze, "threshold": thr,
            "val_auc": auc(yva, pva), "events": events, "history": hist}
    (out / "fold.json").write_text(json.dumps(info, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    log(f"{tag}: лучший val AUC {info['val_auc']}, test AUC {auc(yte, pte)}, порог {thr:.3f}")


def balanced_pick(it, y, n, rng):
    k = n // 2
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    pick = np.concatenate([rng.choice(pos, min(k, len(pos)), replace=False),
                           rng.choice(neg, min(k, len(neg)), replace=False)])
    return it[pick], y[pick]


@torch.no_grad()
def all_feats(model, dl, device, dt):
    train_mode(model, False)
    out = {}
    for x, _ in dl:
        with torch.autocast(device.type, dtype=dt or torch.float32, enabled=dt is not None):
            f = model.all_features(x.to(device, non_blocking=True))
        for k, v in f.items():
            out.setdefault(k, []).append(v.float().cpu())
    return {k: torch.cat(v) for k, v in out.items()}


def probe_head(ftr, ytr, fva, yva, device, kind="linear", norm="bn", steps=300, seed=0):
    """Линейный зонд: голова на замороженных признаках, полный батч, Adam 1e-2 (выбрано автором)."""
    torch.manual_seed(seed)
    head = make_head(ftr.shape[1], kind, 0.0, norm).to(device)
    opt = torch.optim.Adam(head.parameters(), 1e-2, weight_decay=1e-4)
    xt, yt = ftr.to(device), torch.as_tensor(ytr, device=device).long()
    for _ in range(steps):
        loss = F.cross_entropy(head(xt), yt)
        opt.zero_grad()
        loss.backward()
        opt.step()
    head.eval()
    with torch.no_grad():
        pt = torch.softmax(head(xt), -1)[:, 1].cpu().numpy()
        pv = torch.softmax(head(fva.to(device)), -1)[:, 1].cpu().numpy() if fva is not None else None
    return auc(ytr, pt), (auc(yva, pv) if pv is not None else None)


def feature_ablation(a, cl, norm, tr_it, ytr, va_it, yva, device, log, tag):
    """Признаки из замороженного предобученного BENDR → линейный зонд для каждого варианта.
    Возвращает {признак: (train AUC, held-out AUC)}."""
    model = BENDR("ctx_last", dropout=0.0)
    model.load_pretrained(a.encoder, a.contextualizer, lambda *_: None)
    set_freeze(model, "backbone", 0)
    model.to(device)
    dt = amp_dtype(a, device)
    mk = lambda it: DataLoader(Windows(cl, it, norm), batch_size=a.batch_size, num_workers=a.workers)
    t0 = time.time()
    ftr, fva = all_feats(model, mk(tr_it), device, dt), all_feats(model, mk(va_it), device, dt)
    del model
    res = {}
    log(f"{tag}: зонд признаков (замороженный BENDR, линейная голова), train {len(ytr)} / held-out {len(yva)} окон")
    for k in FEATURES:
        rel = float(ftr[k].std(0).mean() / (ftr[k].abs().mean() + 1e-12))
        res[k] = probe_head(ftr[k], ytr, fva[k], yva, device, a.head, "bn")
        log(f"  {k:12s} dim {ftr[k].shape[1]:5d} | разброс {rel:.4f} | train AUC {res[k][0]:.3f} | held-out AUC {res[k][1]:.3f}")
    log(f"{tag}: зонд за {time.time() - t0:.0f} с")
    return res, ftr


FINETUNE_FEATURE = "ctx_mean"


def auto_feature(a, res, log, tag):
    """--feature auto. Зонд меряет только замороженные признаки, поэтому решает лишь при --freeze backbone.
    При дообучении (encoder/partial/full) берётся признак контекстуализатора ctx_mean: учатся слои трансформера,
    а не зондированные признаки (BENDR использует ctx_last; ctx_mean у вас на зонде лучше — выбрано автором)."""
    if a.freeze != "backbone" or res is None:
        log(f"{tag}: --feature auto → {FINETUNE_FEATURE} (freeze {a.freeze}: дообучение, признак контекстуализатора)")
        return FINETUNE_FEATURE
    best = max((k for k in res if k != "ctx_first"), key=lambda k: res[k][1] or 0)
    log(f"{tag}: --feature auto → {best} (freeze backbone: лучший held-out AUC зонда {res[best][1]:.3f})")
    if res[best][1] is not None and res[best][1] < 0.65:
        log(f"  ВНИМАНИЕ: замороженные признаки почти не разделяют классы (held-out AUC < 0.65) — "
            f"ожидайте слабый результат backbone; нужен partial/full.")
    return best


def choose_feature(a, caches, fold, device, log):
    """--feature auto для обучения: при backbone — зонд на первом фолде (train → val), иначе ctx_mean."""
    if a.freeze != "backbone":
        return auto_feature(a, None, log, "AUTO")
    names = list(caches)
    cl = [caches[n] for n in names]
    lab = lambda it: np.array([cl[c].labels[i] for c, i in it], np.int64)
    rng = np.random.default_rng(a.seed)
    tr_it, ytr = balanced_pick(fold["train"], lab(fold["train"]), a.probe_n, rng)
    va_it, yva = balanced_pick(fold["val"], lab(fold["val"]), a.probe_n // 2, rng)
    res, _ = feature_ablation(a, cl, make_norm(a, fold), tr_it, ytr, va_it, yva, device, log, "AUTO")
    return auto_feature(a, res, log, "AUTO")


def sanity_overfit(a, caches, device, log):
    """Проверка, что модель вообще может учиться: --sanity-n сбалансированных окон первого фолда,
    обучение и оценка НА ЭТИХ ЖЕ окнах (val = train). Условия — чистое переобучение:
    CE, без dropout и weight decay, постоянный lr (без затухания), маленький батч, много шагов.
    Перед этим — зонд признаков (какой вход головы вообще разделяет классы). Порог успеха — AUC > 0,9."""
    names = list(caches)
    cl = [caches[n] for n in names]
    lab = lambda it: np.array([cl[c].labels[i] for c, i in it], np.int64)
    fold = make_folds(a, caches, np.random.default_rng(a.seed))[0]
    rng = np.random.default_rng(a.seed)
    it, ys = balanced_pick(fold["train"], lab(fold["train"]), a.sanity_n, rng)
    va_it, yva = balanced_pick(fold["val"], lab(fold["val"]), a.sanity_n, rng)
    norm = make_norm(a, fold)
    feats = res = None
    if a.sanity_ablation or (a.feature == "auto" and a.freeze == "backbone"):
        res, feats = feature_ablation(a, cl, norm, it, ys, va_it, yva, device, log, "SANITY")
    if a.feature == "auto":
        a.feature = auto_feature(a, res, log, "SANITY")
    s = argparse.Namespace(**vars(a))
    s.balancing, s.loss, s.dropout, s.head_dropout = "none", "ce", 0.0, 0.0
    s.weight_decay, s.warmup, s.min_lr_ratio, s.unfreeze_epoch, s.monitor = 0.0, 0.0, 1.0, 0, "auc"
    s.lr_head, s.lr_backbone, s.batch_size = a.sanity_lr_head, a.sanity_lr, a.sanity_batch
    ds = Windows(cl, it, norm)
    model, n_tr = build_model(s, device, log, True)
    dt = amp_dtype(a, device)
    fe = a.freeze == "backbone"   # магистраль заморожена — признаки считаются один раз
    if fe:
        f = feats[s.feature] if feats is not None else extract(model, DataLoader(ds, a.batch_size), device, dt)
        ds = TensorDataset(f, torch.as_tensor(ys))
    tl = DataLoader(ds, batch_size=s.batch_size, shuffle=True, num_workers=0 if fe else a.workers)
    el = DataLoader(ds, batch_size=a.batch_size, num_workers=0 if fe else a.workers)   # val = те же окна
    log(f"SANITY: {len(ys)} окон ({ys.sum()} приступ), val = те же окна; freeze {a.freeze}, обучаемых {n_tr:,}, "
        f"признак {s.feature}, AMP {dt}; CE, dropout 0, wd 0, lr голова {s.lr_head:g} / магистраль {s.lr_backbone:g} "
        f"постоянный, батч {s.batch_size}, эпох {a.sanity_epochs} ({a.sanity_epochs * len(tl)} шагов)")
    if not fe:
        input_stats(tl, log, "SANITY")
    ckpt = Path(a.out) / "sanity_best.pt"
    hist = fit(model, tl, el, ys, Criterion("ce"), s, device, dt, ckpt, log, from_feats=fe,
               epochs=a.sanity_epochs, patience=a.sanity_epochs)
    # Вердикт по ПОСЛЕДНИМ эпохам (текущая модель, eval-режим), а не по лучшей: случайный удачный
    # всплеск AUC на фоне схлопнувшихся предсказаний не должен давать OK.
    last = hist[-5:]
    aucs = [h["auc"] or 0 for h in last]
    fin = hist[-1]
    checks = {
        f"AUC последних {len(last)} эпох ≥ 0.9 (мин {min(aucs):.3f}, среднее {np.mean(aucs):.3f})": min(aucs) >= 0.9,
        f"итоговый val loss < 0.6 и конечен ({fin['val_loss']:.3f})": bool(np.isfinite(fin["val_loss"]) and fin["val_loss"] < 0.6),
        f"предсказания не схлопнулись (p std {fin['p_std']:.3f} > 0.05)": fin["p_std"] > 0.05,
        f"train loss упал ({hist[0]['train_loss']:.3f} → {fin['train_loss']:.3f})": fin["train_loss"] < 0.7 * hist[0]["train_loss"],
    }
    for k, v in checks.items():
        log(f"  {'✓' if v else '✗'} {k}")
    ok = all(checks.values())
    log(f"SANITY: {'OK, модель учится и eval-режим согласован с train' if ok else 'НЕ ПРОЙДЕНО (см. ✗ выше)'}")
    ckpt.unlink(missing_ok=True)
    return ok


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
    ap.add_argument("--freeze", choices=["full", "encoder", "partial", "backbone"], default="full",
                    help="full — всё; encoder — заморожен свёрточный энкодер; partial — учатся последние "
                         "--unfreeze-last слоёв трансформера; backbone — только голова")
    ap.add_argument("--unfreeze-last", type=int, default=2,
                    help="для partial: последние N слоёв трансформера (признак ctx_*) или свёрточных блоков энкодера (enc_*)")
    ap.add_argument("--unfreeze-epoch", type=int, default=0,
                    help="постепенная разморозка: первые N эпох учится только голова (0 — выкл.)")
    ap.add_argument("--fractions", nargs="+", type=float, default=[1.0], help="например 0.25 0.5 1.0")
    ap.add_argument("--balancing", choices=["sampler", "class_weights", "none"], default="sampler",
                    help="ровно один механизм: sampler (WeightedRandomSampler) или веса классов в loss")
    ap.add_argument("--loss", choices=["sswce", "ce", "focal"], default="sswce")
    ap.add_argument("--focal-gamma", type=float, default=2.0)
    ap.add_argument("--norm", choices=["bendr", "global", "window"], default="bendr",
                    help="bendr — min-max окна в [-1,1] + канал SCALE, как при предобучении BENDR (DN3)")
    ap.add_argument("--max-scale", type=float, default=MAX_SCALE, help="размах (В), для канала SCALE при --norm bendr")
    ap.add_argument("--feature", choices=["auto", *FEATURES], default="auto",
                    help="вход головы; auto — при backbone лучший по held-out AUC зонда, при дообучении ctx_mean; "
                         "ctx_last — как BENDRClassification, enc_pool4 — как LinearHeadBENDR; ctx_first почти постоянен")
    ap.add_argument("--head", choices=["linear", "mlp"], default="linear")
    ap.add_argument("--head-norm", choices=["bn", "fixed", "ln"], default="bn",
                    help="нормировка входа головы: bn — BatchNorm, статистика пересчитывается перед каждой оценкой "
                         "(--bn-recal), рост нормы признаков сдерживает --feat-reg; fixed — стандартизация, "
                         "зафиксированная по предобученным признакам (смещение логитов неустойчиво); ln — LayerNorm")
    ap.add_argument("--feat-reg", type=float, default=1.0,
                    help="λ штрафа λ·log²(|признак| / |признак предобученной модели|) при обучении магистрали (0 — выкл.)")
    ap.add_argument("--bn-recal", type=int, default=50,
                    help="батчей train для пересчёта статистики BatchNorm головы перед оценкой (0 — выкл.)")
    ap.add_argument("--head-dropout", type=float, default=0.0)
    ap.add_argument("--dropout", type=float, default=0.0, help="dropout энкодера и трансформера (в дообучении BENDR 0)")
    ap.add_argument("--probe-n", type=int, default=2000, help="окон train для зонда --feature auto")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--patience", type=int, default=15, help="ранняя остановка по --monitor")
    ap.add_argument("--monitor", choices=["auc", "loss"], default="auc", help="критерий лучшей эпохи на валидации")
    ap.add_argument("--batch-size", type=int, default=64, help="BENDR дообучался батчами 4–80")
    ap.add_argument("--lr-head", type=float, default=1e-3, help="lr головы")
    ap.add_argument("--lr-backbone", type=float, default=1e-5,
                    help="lr энкодера и контекстуализатора (BENDR дообучался с 1e-5…5e-5); после --unfreeze-epoch "
                         "разогревается заново --warmup эпох")
    ap.add_argument("--warmup", type=float, default=1.0, help="эпох линейного разогрева lr, далее косинус")
    ap.add_argument("--min-lr-ratio", type=float, default=0.05, help="косинус затухает до lr·ratio, а не до 0")
    ap.add_argument("--clip", type=float, default=1.0, help="клиппинг нормы градиента")
    ap.add_argument("--weight-decay", type=float, default=0.01, help="weight decay головы")
    ap.add_argument("--wd-backbone", type=float, default=0.05,
                    help="weight decay матриц весов магистрали (bias и нормировки без decay)")
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--test-fraction", type=float, default=0.3, help="для patient_wise (выбрано автором)")
    ap.add_argument("--max-bg", type=int, default=None, help="урезать фон в обучающих файлах до N окон")
    ap.add_argument("--samples-per-epoch", type=int, default=None,
                    help="окон за эпоху (sampler с возвращением); короче эпоха — чаще оценка, напр. 100000")
    ap.add_argument("--val-max", type=int, default=None,
                    help="ежепоэпохная валидация на не более N окнах (все приступы + случайный фон), напр. 30000")
    ap.add_argument("--threshold", choices=["youden", "argmax"], default="youden")
    ap.add_argument("--vote-widths", nargs="+", type=int, default=[1, 3, 5, 7], help="1 = без пост-обработки")
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    ap.add_argument("--amp", choices=["auto", "bf16", "fp16", "off"], default="auto",
                    help="auto — bf16, если GPU поддерживает, иначе fp16 с GradScaler; off — всё в fp32")
    ap.add_argument("--amp-scope", choices=["all", "encoder"], default="all",
                    help="encoder — AMP только для свёрточного энкодера, трансформер в fp32")
    ap.add_argument("--diag", action=argparse.BooleanOptionalAction, default=True,
                    help="строка диагностики после каждой эпохи (признаки, bf16/fp32, train/eval)")
    ap.add_argument("--sanity-overfit", action="store_true",
                    help="быстрая проверка: переобучение на --sanity-n окнах, train AUC должен быть > 0,9")
    ap.add_argument("--sanity-n", type=int, default=400)
    ap.add_argument("--sanity-epochs", type=int, default=60)
    ap.add_argument("--sanity-batch", type=int, default=32)
    ap.add_argument("--sanity-lr", type=float, default=1e-4, help="lr магистрали в sanity (постоянный)")
    ap.add_argument("--sanity-lr-head", type=float, default=3e-3, help="lr головы в sanity (постоянный)")
    ap.add_argument("--sanity-ablation", action=argparse.BooleanOptionalAction, default=True,
                    help="в sanity сначала зонд всех вариантов --feature")
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
    if a.sanity_overfit:
        sys.exit(0 if sanity_overfit(a, caches, device, log) else 1)
    folds = make_folds(a, caches, np.random.default_rng(a.seed))
    if a.feature == "auto":
        a.feature = choose_feature(a, caches, folds[0], device, log)
    (run_dir / "args.json").write_text(json.dumps(vars(a), ensure_ascii=False, indent=1), encoding="utf-8")
    first = True
    for fold in folds:
        for frac in a.fractions:
            train_fold(fold, frac, caches, a, device, run_dir, log, first)
            first = False
        fold.pop("feats", None)
    evaluate(run_dir, a.vote_widths)


if __name__ == "__main__":
    main()
