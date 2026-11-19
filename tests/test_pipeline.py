"""Тесты сопоставления каналов, фильтра, окон, нормализации и LOPO-CV."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pyedflib
import pytest
import yaml

from src.preprocessing.annotations import SeizureInterval
from src.preprocessing.channel_mapping import MappingError, map_channels
from src.preprocessing.filtering import FilterError, bandpass_filter
from src.preprocessing.normalization import Normalizer
from src.preprocessing.segmentation import SegmentationError, make_windows
from src.training.lopo_cv import LopoSplitError, collect_windows, generate_lopo_splits

ROOT = Path(__file__).resolve().parents[1]
CHB_NAMES = (
    "FP1-F7",
    "F7-T7",
    "T7-P7",
    "P7-O1",
    "FP1-F3",
    "F3-C3",
    "C3-P3",
    "P3-O1",
    "FP2-F4",
    "F4-C4",
    "C4-P4",
    "P4-O2",
    "FP2-F8",
    "F8-T8",
    "T8-P8",
    "P8-O2",
    "FZ-CZ",
    "CZ-PZ",
    "P7-T7",
    "T7-FT9",
    "FT9-FT10",
    "FT10-T8",
    "T8-P8",
)


def _config() -> dict:
    return yaml.safe_load((ROOT / "configs" / "default.yaml").read_text(encoding="utf-8"))


def _signal() -> np.ndarray:
    rows = []
    for index in range(len(CHB_NAMES)):
        rows.append(np.full(32, index, dtype=np.float32))
    rows[14] = np.full(32, 100.0, dtype=np.float32)
    rows[22] = np.full(32, 999.0, dtype=np.float32)
    return np.stack(rows, axis=0)


def test_map_channels_drops_extras_and_second_duplicate() -> None:
    """Из 23 каналов chb01 остаётся 20, второй T8-P8 не используется."""
    mapped = map_channels(_signal(), CHB_NAMES, _config())
    mapping = _config()["data"]["channels"]["mapping"]

    assert mapped.shape == (20, 32)
    assert mapped.dtype == np.float32
    by_name = {name: _signal()[index] for index, name in enumerate(CHB_NAMES)}
    for row, slot in enumerate(_config()["data"]["channels"]["bendr_order"]):
        source = mapping[slot]
        expected = by_name[source]
        if source == "T8-P8":
            expected = np.full(32, 100.0, dtype=np.float32)
        np.testing.assert_array_equal(mapped[row], expected)
    assert "FT9-FT10" not in mapping.values()
    assert not np.any(mapped == 999.0)


def test_map_channels_missing_slot_raises() -> None:
    config = _config()
    config["data"]["channels"]["mapping"]["SCALE"] = "NO-SUCH"
    with pytest.raises(MappingError, match="SCALE"):
        map_channels(_signal(), CHB_NAMES, config)


def test_map_channels_missing_slot_can_be_zero() -> None:
    config = _config()
    config["data"]["channels"]["on_missing"] = "zero"
    config["data"]["channels"]["mapping"]["SCALE"] = "NO-SUCH"
    mapped = map_channels(_signal(), CHB_NAMES, config)
    np.testing.assert_array_equal(mapped[-1], np.zeros(32, dtype=np.float32))


def test_bandpass_keeps_10hz_and_rejects_100hz() -> None:
    """10 Гц проходит полосу 0.5–50 Гц, 100 Гц подавляется."""
    sfreq = 256.0
    duration = 8.0
    time = np.arange(int(sfreq * duration)) / sfreq
    data = np.vstack(
        [
            np.sin(2 * np.pi * 10 * time),
            np.sin(2 * np.pi * 100 * time),
        ]
    )
    filtered = bandpass_filter(data, sfreq, _config())
    middle = slice(int(sfreq), -int(sfreq))
    kept = np.std(filtered[0, middle])
    removed = np.std(filtered[1, middle])

    assert filtered.shape == data.shape
    assert kept > 0.4
    assert removed < 0.05


def test_filter_rejects_wrong_sampling_rate() -> None:
    with pytest.raises(FilterError, match="не совпадает"):
        bandpass_filter(np.zeros((1, 2048), dtype=np.float32), 128.0, _config())


def test_segment_labels_seizure_and_drops_postictal() -> None:
    """Окно приступа остаётся с меткой 1, 4 секунды фона после него выбрасываются."""
    config = _config()
    config["data"]["postictal_exclusion_min"] = 4 / 60
    sfreq = 256.0
    window = 1024
    data = np.ones((20, window * 4), dtype=np.float32)
    seizures = [SeizureInterval(index=1, start_sec=4.0, end_sec=8.0)]

    windows, labels = make_windows(data, sfreq, seizures, config)

    assert windows.shape == (3, 20, window)
    np.testing.assert_array_equal(labels, np.array([0, 1, 0], dtype=np.int64))


def test_segment_keeps_later_seizure_inside_postictal_gap() -> None:
    """Приступ, начавшийся в постиктальном окне предыдущего, не выбрасывается."""
    config = _config()
    config["data"]["postictal_exclusion_min"] = 8 / 60
    sfreq = 256.0
    window = 1024
    data = np.ones((20, window * 4), dtype=np.float32)
    seizures = [
        SeizureInterval(index=1, start_sec=0.0, end_sec=4.0),
        SeizureInterval(index=2, start_sec=8.0, end_sec=12.0),
    ]

    windows, labels = make_windows(data, sfreq, seizures, config)

    np.testing.assert_array_equal(labels, np.array([1, 1], dtype=np.int64))
    assert windows.shape[0] == 2


def test_segment_keeps_short_seizure_that_spills_into_postictal() -> None:
    """Короткий приступ в начале окна не выбрасывается вместе с постиктальным хвостом."""
    config = _config()
    sfreq = 256.0
    window = 1024
    data = np.ones((20, window), dtype=np.float32)
    seizures = [SeizureInterval(index=1, start_sec=0.0, end_sec=2.0)]

    windows, labels = make_windows(data, sfreq, seizures, config)

    assert windows.shape == (1, 20, window)
    np.testing.assert_array_equal(labels, np.array([1], dtype=np.int64))


def test_segment_rejects_wrong_sampling_rate() -> None:
    data = np.ones((20, 2048), dtype=np.float32)
    with pytest.raises(SegmentationError, match="не совпадает"):
        make_windows(data, 128.0, [], _config())


def test_normalizer_uses_only_train_statistics() -> None:
    """Тестовые нули сдвигаются средним обучения, а не своим средним."""
    config = _config()
    train = np.zeros((4, 2, 8), dtype=np.float32)
    train[:, 0, :] = 10.0
    train[:, 1, :] = 2.0
    test = np.zeros((3, 2, 8), dtype=np.float32)

    normalizer = Normalizer(config)
    transformed_train = normalizer.fit_transform(train)
    transformed_test = normalizer.transform(test)

    np.testing.assert_allclose(transformed_train.mean(axis=(0, 2)), 0.0, atol=1e-6)
    np.testing.assert_allclose(transformed_test[:, 0], -10.0, atol=1e-6)
    np.testing.assert_allclose(transformed_test[:, 1], -2.0, atol=1e-6)
    assert transformed_test.shape == test.shape


def test_normalizer_minmax_scales_train_range() -> None:
    config = _config()
    config["data"]["normalization"]["method"] = "min-max"
    train = np.zeros((2, 1, 4), dtype=np.float32)
    train[0] = 0.0
    train[1] = 10.0
    normalizer = Normalizer(config)
    scaled = normalizer.fit_transform(train)

    assert scaled.min() == pytest.approx(0.0)
    assert scaled.max() == pytest.approx(1.0)
    held_out = normalizer.transform(np.full((1, 1, 4), 5.0, dtype=np.float32))
    np.testing.assert_allclose(held_out, 0.5)


def test_lopo_split_is_stratified_and_leaves_patient_out() -> None:
    config = _config()
    labels = {
        "chb01": np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1]),
        "chb02": np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]),
        "chb03": np.array([0, 0, 1, 1]),
    }
    folds = generate_lopo_splits(["chb01", "chb02", "chb03"], config, labels)

    assert len(folds) == 3
    held_out = folds[2]
    train, val, test = held_out
    assert held_out.test_patient == "chb03"
    assert set(test.patient_ids) == {"chb03"}
    np.testing.assert_array_equal(test.labels, labels["chb03"])
    assert set(train.patient_ids).isdisjoint({"chb03"})
    assert set(val.patient_ids).isdisjoint({"chb03"})
    assert len(train) + len(val) == 20
    assert abs(len(val) / 20 - 0.2) <= 0.05
    assert set(np.unique(val.labels)) == {0, 1}
    assert set(np.unique(train.labels)) == {0, 1}

    stored = {
        "chb01": np.arange(10, dtype=np.float32).reshape(10, 1, 1),
        "chb02": np.arange(10, dtype=np.float32).reshape(10, 1, 1),
        "chb03": np.arange(4, dtype=np.float32).reshape(4, 1, 1),
    }
    collected = collect_windows(stored, test)
    np.testing.assert_array_equal(collected.reshape(-1), np.arange(4))


def test_lopo_rejects_single_class_pool() -> None:
    config = _config()
    labels = {
        "chb01": np.array([0, 0, 0, 0]),
        "chb02": np.array([0, 0, 0, 0]),
    }
    with pytest.raises(LopoSplitError, match="один класс"):
        generate_lopo_splits(["chb01", "chb02"], config, labels)


def _write_edf(path: Path, signals: np.ndarray, labels: list[str], sfreq: int) -> None:
    writer = pyedflib.EdfWriter(str(path), len(labels), file_type=pyedflib.FILETYPE_EDFPLUS)
    try:
        writer.setSignalHeaders(
            [
                {
                    "label": label,
                    "dimension": "uV",
                    "sample_frequency": sfreq,
                    "physical_max": 500.0,
                    "physical_min": -500.0,
                    "digital_max": 32767,
                    "digital_min": -32768,
                    "transducer": "",
                    "prefilter": "",
                }
                for label in labels
            ]
        )
        writer.writeSamples([np.asarray(row, dtype=np.float64) for row in signals])
    finally:
        writer.close()


def test_preprocess_patient_reports_shapes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Сквозной прогон: EDF chb01 превращается в окна [N, 20, 1024]."""
    sfreq = 256
    duration = 8
    time = np.arange(sfreq * duration) / sfreq
    signals = np.vstack([(index + 1) * np.sin(2 * np.pi * 10 * time) for index in range(len(CHB_NAMES))])
    patient_dir = tmp_path / "chb01"
    patient_dir.mkdir()
    _write_edf(patient_dir / "chb01_01.edf", signals, list(CHB_NAMES), sfreq)
    summary = "\n".join(
        [
            "Data Sampling Rate: 256 Hz",
            "Channels in EDF Files:",
            *[f"Channel {index}: {name}" for index, name in enumerate(CHB_NAMES, start=1)],
            "File Name: chb01_01.edf",
            "File Start Time: 11:42:54",
            "File End Time: 11:43:02",
            "Number of Seizures in File: 0",
            "",
        ]
    )
    (patient_dir / "chb01-summary.txt").write_text(summary, encoding="utf-8")

    spec = importlib.util.spec_from_file_location(
        "preprocess_patient",
        ROOT / "scripts" / "preprocess_patient.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    exit_code = module.main(
        ["--all", "--patient", "chb01", "--data-root", str(tmp_path), "--config", str(ROOT / "configs" / "default.yaml")]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Окна [N, C, T]: [2, 20, 1024]" in captured.out
    assert "Класс 0 (фон): 2" in captured.out
    assert "Класс 1 (приступ): 0" in captured.out

    one_file = module.main(
        [
            "--patient",
            "chb01",
            "--edf",
            str(patient_dir / "chb01_01.edf"),
            "--data-root",
            str(tmp_path),
            "--config",
            str(ROOT / "configs" / "default.yaml"),
        ]
    )
    one_captured = capsys.readouterr()
    assert one_file == 0
    assert "Окна [N, C, T]: [2, 20, 1024]" in one_captured.out
    assert "Файлов прочитано: 1" in one_captured.out
