# -*- coding: utf-8 -*-
"""
Назначение: расчёт метрик детекции и сохранение таблиц и рисунков.
Дата: 23.11.2026

Чувствительность, специфичность, FP/ч = (1 − specificity) · 3600 / T_окна,
событийная чувствительность, матрица ошибок, кривые потерь.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .postprocessing import event_sensitivity, group_false_positives, majority_voting


def confusion(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def sensitivity(cm):
    den = cm["tp"] + cm["fn"]
    return cm["tp"] / den if den else None


def specificity(cm):
    den = cm["tn"] + cm["fp"]
    return cm["tn"] / den if den else None


def fp_per_hour(spec, window_sec):
    """Сегментная оценка FP/ч по формуле из диссертации и конфига репозитория."""
    if spec is None:
        return None
    return (1.0 - spec) * 3600.0 / float(window_sec)


def evaluate_sequence(y_true, y_pred, window_sec=4.0, starts=None, seizures=None,
                      vote_width=3, apply_vote=True):
    """Метрики до и после голосования, плюс событийные показатели если есть разметка."""
    y_true = np.asarray(y_true, dtype=np.int64)
    raw = np.asarray(y_pred, dtype=np.int64)
    smooth = majority_voting(raw, vote_width) if apply_vote else raw
    out = {}
    for name, pred in (("raw", raw), ("smooth", smooth)):
        cm = confusion(y_true, pred)
        sen = sensitivity(cm)
        spe = specificity(cm)
        n_neg = cm["tn"] + cm["fp"]
        interictal_hours = n_neg * float(window_sec) / 3600.0
        rec = {
            **cm,
            "sensitivity": sen,
            "specificity": spe,
            "fp_per_hour": fp_per_hour(spe, window_sec),
            "grouped_fp": group_false_positives(pred, y_true),
            "n_background": n_neg,
            "n_seizure": cm["tp"] + cm["fn"],
            "interictal_hours": interictal_hours,
        }
        if interictal_hours > 0:
            rec["grouped_fp_per_hour"] = rec["grouped_fp"] / interictal_hours
        if starts is not None and seizures is not None:
            rec["event_sensitivity"] = event_sensitivity(pred, starts, window_sec, seizures)
        out[name] = rec
    return out


def plot_loss_curves(history, path):
    """history — список словарей с epoch, train_loss, val_loss."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ep = [h["epoch"] for h in history]
    plt.figure(figsize=(6.0, 4.0))
    plt.plot(ep, [h["train_loss"] for h in history], label="обучающая выборка")
    plt.plot(ep, [h["val_loss"] for h in history], label="валидация")
    plt.xlabel("Эпоха")
    plt.ylabel("Функция потерь SSWCE")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_confusion(cm, path, title="Матрица ошибок"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mat = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]], dtype=float)
    plt.figure(figsize=(4.2, 3.6))
    plt.imshow(mat, cmap="Blues")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, int(mat[i, j]), ha="center", va="center", color="black")
    plt.xticks([0, 1], ["фон", "приступ"])
    plt.yticks([0, 1], ["фон", "приступ"])
    plt.xlabel("Прогноз")
    plt.ylabel("Истина")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_table(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    df.to_markdown(path.with_suffix(".md"), index=False)
    return df
