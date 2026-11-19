"""Дешёвые проверки скрипта одной пациента: без полного обучения."""

from pathlib import Path

import numpy as np
import yaml

from scripts.train_single_patient import _split_indices, _try_pretrained, main
from src.models.bendr import BENDRClassifier

ROOT = Path(__file__).resolve().parents[1]


def test_missing_summary_does_not_train(tmp_path):
    code = main(["--patient", "chb01", "--data-root", str(tmp_path), "--epochs", "2"])
    assert code == 2


def test_too_many_epochs_rejected(tmp_path):
    code = main(["--patient", "chb01", "--data-root", str(tmp_path), "--epochs", "4"])
    assert code == 1


def test_split_without_seizures_still_makes_val():
    config = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text(encoding="utf-8"))
    labels = np.zeros(20, dtype=np.int64)
    train_idx, val_idx = _split_indices(labels, config)
    assert len(train_idx) + len(val_idx) == 20
    assert len(val_idx) > 0
    assert len(np.intersect1d(train_idx, val_idx)) == 0


def test_missing_weights_print_warning(capsys, tmp_path):
    config = {
        "data": {"n_channels_out": 20},
        "paths": {"weights_dir": str(tmp_path / "weights")},
        "weights": {"encoder_filename": "encoder.pt"},
        "model": {
            "scale": "tiny",
            "conv_filters": 8,
            "conv_kernels": [3, 2],
            "downsample_factor": 6,
            "group_norm_groups": 4,
            "channel_dropout_p": 0.0,
            "d_model": 16,
            "ff_dim": 32,
            "relative_pos_kernel": 25,
            "relative_pos_groups": 16,
            "masking": False,
            "n_classes": 2,
            "use_first_token": True,
            "scales": {"tiny": {"conv_blocks": 2, "transformer_layers": 1, "transformer_heads": 4}},
        },
    }
    _try_pretrained(BENDRClassifier(config), config)
    assert "Веса не найдены, инициализация случайная" in capsys.readouterr().out
