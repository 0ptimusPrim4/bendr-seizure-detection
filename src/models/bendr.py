"""BENDR: свёртки, трансформер и голова на два класса.

load_pretrained подставляет только энкодер. Голова остаётся случайной.
Маскирование на тонкой настройке выключено.
"""

from pathlib import Path

import torch
from torch import nn

from src.models.layers import ChannelDropout, RelativePosition


class ModelConfigError(ValueError):
    """Конфиг модели кривой."""


class PretrainedLoadError(RuntimeError):
    """Файл весов не подставился в энкодер."""


class ClassificationHead(nn.Module):
    """Linear и Softmax: фон и приступ."""

    def __init__(self, d_model, n_classes):
        super().__init__()
        if d_model < 1 or n_classes < 2:
            raise ModelConfigError(f"Голове нужны d_model и хотя бы 2 класса, пришло {d_model}, {n_classes}.")
        self.linear = nn.Linear(d_model, n_classes)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, token):
        return self.softmax(self.linear(token))


class BENDREncoder(nn.Module):
    """Шесть (или сколько в конфиге) свёрток и трансформер.

    В каждом блоке: Conv1d, ChannelDropout, GroupNorm, GELU.
    Шаг свёртки равен ядру, произведение шагов — downsample_factor.
    """

    def __init__(
        self,
        in_channels,
        conv_filters,
        kernels,
        group_norm_groups,
        channel_dropout_p,
        d_model,
        n_layers,
        n_heads,
        ff_dim,
        relative_pos_kernel=25,
        relative_pos_groups=16,
    ):
        super().__init__()
        if in_channels < 1:
            raise ModelConfigError(f"Число входных каналов должно быть положительным, получено: {in_channels}.")
        if n_layers < 1 or n_heads < 1:
            raise ModelConfigError("У трансформера должны быть слои и головы.")
        if d_model % n_heads != 0:
            raise ModelConfigError(f"d_model={d_model} не делится на число голов {n_heads}.")
        if group_norm_groups < 1 or conv_filters % group_norm_groups != 0:
            raise ModelConfigError(
                f"group_norm_groups={group_norm_groups} не делит число фильтров {conv_filters}."
            )

        self.kernels = tuple(kernels)
        blocks = []
        incoming = in_channels
        for kernel in self.kernels:
            # NOTE: stride = kernel. Произведение [3,2,2,2,2,2] как раз 96, не разъезжать списки
            blocks.append(
                nn.Sequential(
                    nn.Conv1d(incoming, conv_filters, kernel_size=kernel, stride=kernel, bias=False),
                    ChannelDropout(channel_dropout_p),
                    nn.GroupNorm(group_norm_groups, conv_filters),
                    nn.GELU(),
                )
            )
            incoming = conv_filters
        self.frontend = nn.Sequential(*blocks)
        self.projection = nn.Linear(conv_filters, d_model)
        self.start_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.relative_position = RelativePosition(d_model, relative_pos_kernel, relative_pos_groups)
        # TODO: T-fixup из статьи ещё не включён, инициализация пока штатная от PyTorch
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        # FIXME: ChannelDropout стоит после свёртки, так что глушит карты признаков, не сырые электроды

    def forward(self, signal):
        features = self.frontend(signal)
        if features.shape[-1] < 1:
            raise ModelConfigError("Свёртки съели временную ось: увеличьте длину окна или уменьшите шаги.")
        tokens = self.projection(features.transpose(1, 2))
        start = self.start_token.expand(tokens.shape[0], -1, -1)
        tokens = torch.cat((start, tokens), dim=1)
        tokens = tokens + self.relative_position(tokens)
        return self.transformer(tokens)


class BENDRClassifier(nn.Module):
    """Энкодер BENDR и голова из двух классов.

    Вход ``forward`` — батч окон ``[B, C, T]``, для конфига по умолчанию
    ``[B, 20, 1024]``. Выход — вероятности ``[B, 2]``.
    """

    def __init__(self, config: dict) -> None:
        super().__init__()
        model = _model_section(config)
        spec = _architecture(model)
        self.n_channels = _input_channels(config)
        self.use_first_token = bool(model.get("use_first_token", True))
        self.encoder = BENDREncoder(
            in_channels=self.n_channels,
            conv_filters=int(model["conv_filters"]),
            kernels=spec.kernels,
            group_norm_groups=int(model["group_norm_groups"]),
            channel_dropout_p=float(model["channel_dropout_p"]),
            d_model=int(model["d_model"]),
            n_layers=spec.n_layers,
            n_heads=spec.n_heads,
            ff_dim=int(model["ff_dim"]),
            relative_pos_kernel=int(model.get("relative_pos_kernel", 25)),
            relative_pos_groups=int(model.get("relative_pos_groups", 16)),
        )
        self.head = ClassificationHead(int(model["d_model"]), int(model["n_classes"]))
        self._weight_names = _weight_names(config)

    def forward(self, signal):
        """Батч [B, C, T] -> вероятности [B, 2]."""
        if signal.ndim != 3:
            raise ModelConfigError(f"Ожидался батч [B, C, T], получено ndim={signal.ndim}.")
        if signal.shape[1] != self.n_channels:
            raise ModelConfigError(
                f"В батче {signal.shape[1]} каналов, архитектура собрана на {self.n_channels}."
            )
        encoded = self.encoder(signal)
        # первый токен — стартовый. Иначе среднее по всей последовательности
        token = encoded[:, 0] if self.use_first_token else encoded.mean(dim=1)
        return self.head(token)

    def load_pretrained(self, path: str | Path) -> None:
        """Загрузить веса энкодера и оставить голову случайной.

        Args:
            path: Файл ``state_dict`` или каталог с ``encoder.pt`` и
                ``contextualizer.pt`` из раздела ``weights``.

        Raises:
            PretrainedLoadError: Файла нет или ни один тензор не совпал
                с параметрами энкодера.
        """
        weight_path = Path(path)
        if not weight_path.exists():
            raise PretrainedLoadError(
                f"Файл весов не найден: {weight_path}. "
                "Положите релиз v0.1-alpha в weights/: "
                "https://github.com/SPOClab-ca/BENDR/releases/tag/v0.1-alpha"
            )
        incoming = _read_checkpoint(weight_path, self._weight_names)
        encoder_state = _encoder_tensors(incoming)
        current = self.encoder.state_dict()
        matched = {key: value for key, value in encoder_state.items() if key in current}
        if not matched:
            sample = ", ".join(list(encoder_state)[:5]) or "пусто"
            raise PretrainedLoadError(
                "В файле нет тензоров энкодера с известными именами. "
                f"Первые ключи: {sample}."
            )
        current.update(matched)
        self.encoder.load_state_dict(current)


class _Architecture:
    """Число блоков выбранного масштаба и список ядер."""

    def __init__(self, kernels: tuple[int, ...], n_layers: int, n_heads: int) -> None:
        self.kernels = kernels
        self.n_layers = n_layers
        self.n_heads = n_heads


def _model_section(config: dict) -> dict:
    if not isinstance(config, dict):
        raise ModelConfigError("Конфиг модели должен быть словарём.")
    if "d_model" in config and "conv_filters" in config:
        return config
    model = config.get("model")
    if isinstance(model, dict):
        return model
    raise ModelConfigError("В конфиге нет раздела model.")


def _architecture(model: dict) -> _Architecture:
    if bool(model.get("masking", False)):
        raise ModelConfigError("На тонкой настройке model.masking должен быть false.")
    scale_name = str(model["scale"])
    scales = dict(model["scales"])
    if scale_name not in scales:
        raise ModelConfigError(f"Неизвестный масштаб модели: {scale_name}.")
    chosen = dict(scales[scale_name])
    missing_scale = (
        chosen.get("conv_blocks") is None
        or chosen.get("transformer_layers") is None
        or chosen.get("transformer_heads") is None
    )
    if missing_scale:
        raise ModelConfigError(f"Масштаб {scale_name} в конфиге не заполнен.")
    n_blocks = int(chosen["conv_blocks"])
    kernels = tuple(int(kernel) for kernel in model["conv_kernels"])
    if len(kernels) != n_blocks:
        raise ModelConfigError(
            f"conv_kernels содержит {len(kernels)} значений, а масштаб {scale_name} задаёт {n_blocks} блоков."
        )
    if any(kernel < 1 for kernel in kernels):
        raise ModelConfigError("Ядра свёртки должны быть положительными.")
    downsample = 1
    for kernel in kernels:
        downsample *= kernel
    expected = int(model["downsample_factor"])
    if downsample != expected:
        raise ModelConfigError(
            f"Произведение шагов свёртки равно {downsample}, а downsample_factor = {expected}."
        )
    return _Architecture(kernels, int(chosen["transformer_layers"]), int(chosen["transformer_heads"]))


def _input_channels(config: dict) -> int:
    data = config.get("data")
    if isinstance(data, dict) and "n_channels_out" in data:
        return int(data["n_channels_out"])
    channels = config.get("channels")
    if isinstance(channels, dict) and "bendr_order" in channels:
        return len(tuple(channels["bendr_order"]))
    if isinstance(data, dict):
        nested = data.get("channels")
        if isinstance(nested, dict) and "bendr_order" in nested:
            return len(tuple(nested["bendr_order"]))
    if "n_channels" in config:
        return int(config["n_channels"])
    raise ModelConfigError("Не задано число входных каналов: нужен data.n_channels_out.")


def _weight_names(config: dict) -> tuple[str, str]:
    weights = config.get("weights")
    if not isinstance(weights, dict):
        return ("encoder.pt", "contextualizer.pt")
    return (str(weights.get("encoder_filename", "encoder.pt")), str(weights.get("contextualizer_filename", "contextualizer.pt")))


def _read_checkpoint(path: Path, filenames: tuple[str, str]) -> dict[str, torch.Tensor]:
    if path.is_file():
        return _unwrap(torch.load(path, map_location="cpu", weights_only=True))
    files = [path / name for name in filenames if (path / name).is_file()]
    if not files:
        raise PretrainedLoadError(
            f"В каталоге {path} нет {filenames[0]} или {filenames[1]}."
        )
    merged: dict[str, torch.Tensor] = {}
    for file_path in files:
        merged.update(_unwrap(torch.load(file_path, map_location="cpu", weights_only=True)))
    return merged


def _unwrap(loaded: object) -> dict[str, torch.Tensor]:
    if isinstance(loaded, dict) and "state_dict" in loaded and isinstance(loaded["state_dict"], dict):
        loaded = loaded["state_dict"]
    if not isinstance(loaded, dict):
        raise PretrainedLoadError("Чекпоинт должен быть словарём тензоров.")
    tensors: dict[str, torch.Tensor] = {}
    for key, value in loaded.items():
        if torch.is_tensor(value):
            tensors[str(key)] = value
    return tensors


def _encoder_tensors(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Убрать префиксы обёртки и выкинуть веса головы."""
    cleaned: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        name = key
        for prefix in ("module.", "model.", "encoder."):
            if name.startswith(prefix):
                name = name[len(prefix):]
        if name.startswith("head.") or name.startswith("classifier."):
            continue
        cleaned[name] = value
    return cleaned
