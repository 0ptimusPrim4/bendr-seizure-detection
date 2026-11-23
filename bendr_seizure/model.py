# -*- coding: utf-8 -*-
"""
Назначение: классификатор BENDR, совместимый с официальными весами v0.1-alpha.
Дата: 23.11.2026

Логика тонкой настройки сохранена из репозитория пользователя
(голова на два класса, первый токен, без маскирования).
Свёрточный энкодер и контекстуализатор приведены к структуре SPOClab-ca/BENDR,
иначе encoder.pt и contextualizer.pt не загружаются: в официальном коде
чётные ширины ядер увеличиваются на 1, у свёртки есть padding и bias,
проекция 512→1536 находится в контекстуализаторе, а не в линейном слое энкодера.
"""

from pathlib import Path
import torch
from torch import nn


class ChannelDropout(nn.Module):
    """С вероятностью p зануляет канал целиком (аналог Dropout2d из BENDR)."""

    def __init__(self, p):
        super().__init__()
        self.p = float(p)

    def forward(self, x):
        if not self.training or self.p == 0.0:
            return x
        keep = torch.bernoulli(x.new_full((x.shape[0], x.shape[1], 1), 1.0 - self.p))
        return x * keep


class _Hax(nn.Module):
    """Замена LayerNorm в слоях трансформера (T-Fixup, как в официальном коде)."""

    def forward(self, x):
        return x


class Permute(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = tuple(dims)

    def forward(self, x):
        return x.permute(*self.dims)


class ConvEncoderBENDR(nn.Module):
    """Шесть блоков Conv1d–Dropout–GroupNorm–GELU, как ConvEncoderBENDR."""

    def __init__(self, in_channels, encoder_h, enc_width, enc_downsample, dropout_p):
        super().__init__()
        # Официальный приём: ядро делают нечётным, чтобы работал симметричный padding.
        widths = [w if w % 2 else w + 1 for w in enc_width]
        self.encoder = nn.Sequential()
        incoming = in_channels
        for i, (width, stride) in enumerate(zip(widths, enc_downsample)):
            self.encoder.add_module(
                f"Encoder_{i}",
                nn.Sequential(
                    nn.Conv1d(incoming, encoder_h, width, stride=stride, padding=width // 2),
                    ChannelDropout(dropout_p),
                    nn.GroupNorm(encoder_h // 2, encoder_h),
                    nn.GELU(),
                ),
            )
            incoming = encoder_h

    def forward(self, x):
        return self.encoder(x)


class BENDRContextualizer(nn.Module):
    """Контекстуализатор: относительные позиции, проекция в 1536 и 8 слоёв трансформера."""

    def __init__(self, encoder_h, d_model, ff_dim, n_layers, n_heads,
                 pos_kernel, pos_groups, dropout, start_token=-5.0):
        super().__init__()
        self.start_token = start_token
        conv = nn.Conv1d(encoder_h, encoder_h, pos_kernel, padding=pos_kernel // 2, groups=pos_groups)
        conv = nn.utils.weight_norm(conv, dim=2)
        self.relative_position = nn.Sequential(conv, nn.GELU())
        self.input_conditioning = nn.Sequential(
            Permute([0, 2, 1]),
            nn.LayerNorm(encoder_h),
            nn.Dropout(dropout),
            Permute([0, 2, 1]),
            nn.Conv1d(encoder_h, d_model, 1),
            Permute([2, 0, 1]),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=ff_dim,
            dropout=dropout, activation="gelu", batch_first=False,
        )
        layer.norm1 = _Hax()
        layer.norm2 = _Hax()
        self.transformer_layers = nn.ModuleList([
            type(layer)(
                d_model=d_model, nhead=n_heads, dim_feedforward=ff_dim,
                dropout=dropout, activation="gelu", batch_first=False,
            )
            for _ in range(n_layers)
        ])
        for blk in self.transformer_layers:
            blk.norm1 = _Hax()
            blk.norm2 = _Hax()
        self.norm = nn.LayerNorm(d_model)
        self.mask_replacement = nn.Parameter(torch.zeros(encoder_h))
        self.output_layer = nn.Conv1d(d_model, encoder_h, 1)

    def forward(self, x):
        # x: [B, 512, T]
        x = x + self.relative_position(x)
        x = self.input_conditioning(x)  # [T, B, 1536]
        if self.start_token is not None:
            token = x.new_full((1, x.shape[1], x.shape[2]), float(self.start_token))
            x = torch.cat([token, x], dim=0)
        for layer in self.transformer_layers:
            x = layer(x)
        return self.output_layer(x.permute(1, 2, 0))  # [B, 512, T+1]


class BENDRClassifier(nn.Module):
    """Энкодер + контекстуализатор + линейная голова на два класса."""

    def __init__(self, cfg):
        super().__init__()
        m = cfg["model"]
        c_in = int(cfg["data"]["n_channels_out"])
        self.encoder_h = int(m["encoder_h"])
        self.use_first_token = bool(m.get("use_first_token", True))
        self.encoder = ConvEncoderBENDR(
            c_in, self.encoder_h, m["enc_width"], m["enc_downsample"],
            float(m["channel_dropout_p"]),
        )
        self.contextualizer = BENDRContextualizer(
            self.encoder_h, int(m["d_model"]), int(m["ff_dim"]),
            int(m["n_layers"]), int(m["n_heads"]),
            int(m["relative_pos_kernel"]), int(m["relative_pos_groups"]),
            float(m.get("contextualizer_dropout", 0.15)),
        )
        self.head = nn.Sequential(
            nn.Linear(self.encoder_h, int(m["n_classes"])),
            nn.Softmax(dim=-1),
        )

    def forward(self, signal):
        # signal: [B, C, T]
        encoded = self.encoder(signal)
        context = self.contextualizer(encoded)
        token = context[:, :, 0] if self.use_first_token else context.mean(dim=-1)
        return self.head(token)

    def load_pretrained(self, weights_dir, encoder_name="encoder.pt",
                        contextualizer_name="contextualizer.pt"):
        """Подставляет официальные веса, голову оставляет случайной."""
        folder = Path(weights_dir)
        enc_path = folder / encoder_name
        ctx_path = folder / contextualizer_name
        report = {}
        if enc_path.is_file():
            state = torch.load(enc_path, map_location="cpu", weights_only=True)
            missing, unexpected = self.encoder.load_state_dict(state, strict=False)
            report["encoder"] = {
                "file": str(enc_path),
                "missing": list(missing),
                "unexpected": list(unexpected),
            }
        else:
            raise FileNotFoundError(f"Нет весов энкодера: {enc_path}")
        if ctx_path.is_file():
            state = torch.load(ctx_path, map_location="cpu", weights_only=True)
            missing, unexpected = self.contextualizer.load_state_dict(state, strict=False)
            report["contextualizer"] = {
                "file": str(ctx_path),
                "missing": list(missing),
                "unexpected": list(unexpected),
            }
        else:
            raise FileNotFoundError(f"Нет весов контекстуализатора: {ctx_path}")
        return report
