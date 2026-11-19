"""Лосс с мягкими чувствительностью и специфичностью и один шаг тренера."""

from pathlib import Path

import torch
import yaml

from src.models.bendr import BENDRClassifier
from src.models.layers import ChannelDropout
from src.training.loss import SensitivitySpecificityWeightedCrossEntropy
from src.training.trainer import Trainer

ROOT = Path(__file__).resolve().parents[1]


def _config():
    return yaml.safe_load((ROOT / "configs" / "default.yaml").read_text(encoding="utf-8"))


def test_loss_reads_alpha_beta_and_lets_gradient_through():
    criterion = SensitivitySpecificityWeightedCrossEntropy(_config())
    assert criterion.alpha == 0.3
    assert criterion.beta == 0.5

    good = torch.tensor([[0.9, 0.1], [0.1, 0.9]], requires_grad=True)
    bad = torch.tensor([[0.1, 0.9], [0.9, 0.1]], requires_grad=True)
    labels = torch.tensor([0, 1])

    good_loss = criterion(good, labels)
    bad_loss = criterion(bad, labels)
    assert good_loss < bad_loss

    good_loss.backward()
    assert good.grad is not None
    assert good.grad.abs().sum() > 0


def _tiny_model_config():
    config = _config()
    config["model"] = {
        "scale": "tiny",
        "conv_filters": 8,
        "conv_kernels": [3, 2],
        "downsample_factor": 6,
        "group_norm_groups": 4,
        "channel_dropout_p": 0.0,
        "activation": "gelu",
        "d_model": 16,
        "ff_dim": 32,
        "relative_pos_kernel": 25,
        "relative_pos_groups": 16,
        "masking": False,
        "n_classes": 2,
        "use_first_token": True,
        "scales": {"tiny": {"conv_blocks": 2, "transformer_layers": 1, "transformer_heads": 4}},
    }
    config["device"] = {"prefer_cuda": False}
    return config


def test_conv_blocks_drop_whole_channels_and_epoch_saves_checkpoint(tmp_path):
    config = _tiny_model_config()
    model = BENDRClassifier(config)
    dropouts = [module for module in model.encoder.frontend.modules() if isinstance(module, ChannelDropout)]
    assert len(dropouts) == 2
    assert model.encoder.relative_position.conv.kernel_size == (25,)
    assert model.encoder.relative_position.conv.groups == 16

    trainer = Trainer(model, config)
    windows = torch.randn(4, 20, 32)
    labels = torch.tensor([0, 1, 0, 1])
    before = model.head.linear.weight.detach().clone()
    checkpoint = tmp_path / "best.pt"
    train_loss, val_loss, stop = trainer.run_epoch([(windows, labels)], [(windows, labels)], checkpoint)

    assert train_loss > 0
    assert val_loss > 0
    assert stop is False
    assert checkpoint.is_file()
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert "model" in saved
    assert not torch.equal(before, model.head.linear.weight.detach())
