"""Классификатор BENDR, загрузка весов энкодера и ранняя остановка."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from src.models.bendr import BENDRClassifier, ModelConfigError, PretrainedLoadError, _architecture, _model_section
from src.training.trainer import EarlyStopping

ROOT = Path(__file__).resolve().parents[1]


def _tiny_config() -> dict:
    return {
        "data": {"n_channels_out": 20},
        "model": {
            "scale": "tiny",
            "conv_filters": 8,
            "conv_kernels": [3, 2],
            "downsample_factor": 6,
            "group_norm_groups": 4,
            "channel_dropout_p": 0.0,
            "d_model": 16,
            "ff_dim": 32,
            "masking": False,
            "n_classes": 2,
            "use_first_token": True,
            "scales": {
                "tiny": {"conv_blocks": 2, "transformer_layers": 1, "transformer_heads": 4},
            },
        },
    }


def test_default_architecture_matches_config() -> None:
    """Масштаб 100 % сходится с ядрами и коэффициентом понижения частоты."""
    config = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text(encoding="utf-8"))
    spec = _architecture(_model_section(config))
    assert spec.kernels == (3, 2, 2, 2, 2, 2)
    assert spec.n_layers == 8
    assert spec.n_heads == 8


def test_unspecified_quarter_scale_is_rejected() -> None:
    config = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text(encoding="utf-8"))
    config["model"]["scale"] = "25%"
    with pytest.raises(ModelConfigError, match="не заполнен"):
        _architecture(_model_section(config))


def test_classifier_accepts_bendr_window_length() -> None:
    """Реальные шаги свёртки принимают окно 1024 отсчёта и дают [B, 2]."""
    config = _tiny_config()
    config["model"]["conv_kernels"] = [3, 2, 2, 2, 2, 2]
    config["model"]["downsample_factor"] = 96
    config["model"]["scales"]["tiny"]["conv_blocks"] = 6
    model = BENDRClassifier(config)
    model.eval()
    probabilities = model(torch.zeros(1, 20, 1024))
    assert probabilities.shape == (1, 2)
    assert torch.allclose(probabilities.sum(dim=-1), torch.ones(1), atol=1e-5)


def test_classifier_forward_returns_class_probabilities() -> None:
    """Окно [B, 20, T] становится парой вероятностей."""
    model = BENDRClassifier(_tiny_config())
    model.eval()
    probabilities = model(torch.zeros(2, 20, 32))
    assert probabilities.shape == (2, 2)
    assert torch.allclose(probabilities.sum(dim=-1), torch.ones(2), atol=1e-5)


def test_load_pretrained_keeps_random_head(tmp_path: Path) -> None:
    """Веса энкодера подставляются, голова из чекпоинта игнорируется."""
    source = BENDRClassifier(_tiny_config())
    target = BENDRClassifier(_tiny_config())
    head_before = target.head.linear.weight.detach().clone()
    checkpoint = source.state_dict()
    checkpoint["head.linear.weight"] = torch.full_like(checkpoint["head.linear.weight"], 7.0)
    path = tmp_path / "bendr.pt"
    torch.save(checkpoint, path)

    target.load_pretrained(path)

    assert torch.allclose(target.encoder.projection.weight, source.encoder.projection.weight)
    assert torch.equal(target.head.linear.weight, head_before)
    assert not torch.allclose(target.head.linear.weight, torch.full_like(head_before, 7.0))


def test_load_pretrained_missing_file(tmp_path: Path) -> None:
    model = BENDRClassifier(_tiny_config())
    with pytest.raises(PretrainedLoadError, match="не найден"):
        model.load_pretrained(tmp_path / "absent.pt")


def test_early_stopping_uses_config_patience() -> None:
    """Остановка наступает после заданного числа эпох без улучшения val_loss."""
    config = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text(encoding="utf-8"))
    configured = EarlyStopping(config)
    assert configured.monitor == "val_loss"
    assert configured.patience == 15
    assert configured.min_delta == pytest.approx(1e-4)

    stopper = EarlyStopping(
        {
            "early_stopping_monitor": "val_loss",
            "early_stopping_patience": 2,
            "early_stopping_min_delta": 0.1,
        }
    )
    assert stopper.step({"val_loss": 1.0}) is False
    assert stopper.step({"val_loss": 0.8}) is False
    assert stopper.step({"val_loss": 0.75}) is False
    assert stopper.step({"val_loss": 0.74}) is True
    assert stopper.should_stop is True
