"""Тесты разбора сводок CHB-MIT, чтения EDF и конфига по умолчанию."""

from __future__ import annotations

import math
import os
import textwrap
from pathlib import Path

import numpy as np
import pyedflib
import pytest
import yaml

from src.preprocessing.annotations import (
    AnnotationParseError,
    parse_summary,
    parse_summary_text,
)
from src.preprocessing.edf_loader import EdfLoadError, load_edf

ROOT = Path(__file__).resolve().parents[1]


CHB01_EXCERPT = textwrap.dedent(
    """\
    Data Sampling Rate: 256 Hz
    *************************

    Channels in EDF Files:
    **********************
    Channel 1: FP1-F7
    Channel 2: F7-T7
    Channel 15: T8-P8
    Channel 23: T8-P8

    File Name: chb01_01.edf
    File Start Time: 11:42:54
    File End Time: 12:42:54
    Number of Seizures in File: 0

    File Name: chb01_03.edf
    File Start Time: 13:43:04
    File End Time: 14:43:04
    Number of Seizures in File: 1
    Seizure Start Time: 2996 seconds
    Seizure End Time: 3036 seconds
    """
)


def test_parse_unnumbered_single_seizure() -> None:
    """Одиночный приступ без номера, как в chb01-summary.txt."""
    summary = parse_summary_text(CHB01_EXCERPT, patient_id="chb01", source_name="chb01")

    assert summary.patient_id == "chb01"
    assert summary.sampling_rate_hz == 256
    assert summary.n_files == 2
    assert summary.n_seizure_files == 1
    assert summary.n_seizures == 1

    background, seizure_file = summary.files
    assert background.n_seizures == 0
    assert background.seizures == ()
    assert seizure_file.file_name == "chb01_03.edf"
    assert seizure_file.start_time == "13:43:04"
    assert seizure_file.end_time == "14:43:04"
    assert seizure_file.seizures[0].index == 1
    assert seizure_file.seizures[0].start_sec == 2996
    assert seizure_file.seizures[0].end_sec == 3036
    assert seizure_file.seizures[0].duration_sec == 40
    assert seizure_file.channels == ("FP1-F7", "F7-T7", "T8-P8", "T8-P8")


def test_parse_keeps_duplicate_channel_names() -> None:
    """Повтор T8-P8 — это две строки монтажа, а не ошибка сводки."""
    summary = parse_summary_text(CHB01_EXCERPT, patient_id="chb01")
    assert summary.files[0].channels.count("T8-P8") == 2


def test_parse_unnumbered_repeated_seizures_without_clock() -> None:
    """Формат chb24: несколько приступов без номеров и без часов файла."""
    text = textwrap.dedent(
        """\
        Data Sampling Rate: 256 Hz

        Channels in EDF Files:
        Channel 1: FP1-F7

        File Name: chb24_01.edf
        Number of Seizures in File: 2
        Seizure Start Time: 480 seconds
        Seizure End Time: 505 seconds
        Seizure Start Time: 2451 seconds
        Seizure End Time: 2476 seconds
        """
    )
    summary = parse_summary_text(text, patient_id="chb24")
    record = summary.files[0]

    assert record.start_time is None
    assert record.end_time is None
    assert [(item.index, item.start_sec, item.end_sec) for item in record.seizures] == [
        (1, 480, 505),
        (2, 2451, 2476),
    ]


def test_parse_numbered_seizures_and_channel_change() -> None:
    """Нумерованные интервалы, лишние пробелы и смена монтажа между файлами."""
    text = textwrap.dedent(
        """\
        Channels in EDF Files:
        **********************
        Channel 1: FP1-F7
        Channel 2: F7-T7

        File Name: chb04_05.edf
        File Start Time: 10:02:37
        File End Time: 12:41:33
        Number of Seizures in File: 1
        Seizure Start Time: 7804 seconds
        Seizure End Time: 7853 seconds

        Channels changed:
        *****************
        Channel 1: FP1-F7
        Channel 2: -
        Channel 3: ECG

        File Name: chb12_27.edf
        File Start Time: 16:07:06
        File End Time: 17:07:06
        Number of Seizures in File: 2
        Seizure 1 Start Time: 916 seconds
        Seizure 1 End Time:  951 seconds
        Seizure 2 Start Time:  2388 seconds
        Seizure 2 End Time: 2440 seconds
        """
    )
    summary = parse_summary_text(text, patient_id="chb12")

    first, second = summary.files
    assert first.channels == ("FP1-F7", "F7-T7")
    assert first.seizures[0].start_sec == 7804
    assert second.channels == ("FP1-F7", "-", "ECG")
    assert second.seizures[0].end_sec == 951
    assert second.seizures[1].start_sec == 2388
    assert second.seizures[1].end_sec == 2440


def test_parse_crlf_and_patient_id_from_filename(tmp_path: Path) -> None:
    """Идентификатор пациента берётся из имени chbXX-summary.txt."""
    path = tmp_path / "chb17-summary.txt"
    path.write_bytes(
        "File Name: chb17a_03.edf\r\n"
        "Number of Seizures in File: 1\r\n"
        "Seizure 1 Start Time: 2282 seconds\r\n"
        "Seizure 1 End Time: 2372 seconds\r\n".encode("utf-8")
    )

    summary = parse_summary(path)

    assert summary.patient_id == "chb17"
    assert summary.files[0].file_name == "chb17a_03.edf"
    assert summary.files[0].seizures[0].start_sec == 2282
    assert summary.sampling_rate_hz is None


def test_parse_rejects_count_mismatch() -> None:
    """Заявленное число приступов обязано совпасть с найденными интервалами."""
    text = textwrap.dedent(
        """\
        File Name: chb01_03.edf
        Number of Seizures in File: 2
        Seizure Start Time: 2996 seconds
        Seizure End Time: 3036 seconds
        """
    )
    with pytest.raises(AnnotationParseError, match="заявлено приступов"):
        parse_summary_text(text, patient_id="chb01")


def test_parse_rejects_gap_in_seizure_numbers() -> None:
    """Номера приступов в одном файле идут подряд с единицы."""
    text = textwrap.dedent(
        """\
        File Name: chb06_01.edf
        Number of Seizures in File: 2
        Seizure 1 Start Time: 10 seconds
        Seizure 1 End Time: 20 seconds
        Seizure 3 Start Time: 30 seconds
        Seizure 3 End Time: 40 seconds
        """
    )
    with pytest.raises(AnnotationParseError, match="номера приступов"):
        parse_summary_text(text, patient_id="chb06")


def test_parse_rejects_non_positive_duration() -> None:
    text = textwrap.dedent(
        """\
        File Name: chb01_03.edf
        Number of Seizures in File: 1
        Seizure Start Time: 100 seconds
        Seizure End Time: 100 seconds
        """
    )
    with pytest.raises(AnnotationParseError, match="неположительную длительность"):
        parse_summary_text(text, patient_id="chb01")


def test_parse_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="не найден"):
        parse_summary(tmp_path / "chb01-summary.txt")


def test_default_config_locks_task_values() -> None:
    """В конфиге ровно те гиперпараметры, которые заданы в постановке."""
    config = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text(encoding="utf-8"))

    assert config["data"]["sampling_rate_hz"] == 256
    assert config["data"]["window_sec"] == 4
    assert config["data"]["postictal_exclusion_min"] == 15
    assert config["data"]["patients"] == [f"chb{index:02d}" for index in range(1, 24)]
    assert config["data"]["critical_patients"] == ["chb04", "chb12", "chb15"]
    assert config["data"]["filter"] == {
        "type": "butterworth",
        "order": 5,
        "low_hz": 0.5,
        "high_hz": 50.0,
    }
    mapping = config["data"]["channels"]["mapping"]
    bendr_order = config["data"]["channels"]["bendr_order"]
    assert list(mapping) == bendr_order
    assert len(mapping) == 20
    assert bendr_order[-1] == "SCALE"
    assert config["data"]["channels"]["drop"] == ["FT9-FT10", "FT10-T8"]
    assert config["data"]["channels"]["on_missing"] == "error"
    assert list(mapping.values()).count("T8-P8") == 1
    assert "FT9-FT10" not in mapping.values()
    assert config["data"]["channels"]["chb01_reference"].count("T8-P8") == 2

    assert config["model"]["conv_filters"] == 512
    assert config["model"]["conv_kernels"] == [3, 2, 2, 2, 2, 2]
    assert math.prod(config["model"]["conv_kernels"]) == config["model"]["downsample_factor"] == 96
    assert config["model"]["d_model"] == 1536
    assert config["model"]["ff_dim"] == 3076
    assert config["model"]["channel_dropout_p"] == 0.5
    assert config["model"]["group_norm_groups"] == 256
    assert config["model"]["scale"] == "100%"
    assert config["model"]["scales"]["100%"] == {
        "conv_blocks": 6,
        "transformer_layers": 8,
        "transformer_heads": 8,
    }
    assert config["model"]["scales"]["55%"] == {
        "conv_blocks": 6,
        "transformer_layers": 4,
        "transformer_heads": 4,
    }
    assert config["model"]["scales"]["25%"]["transformer_layers"] is None
    assert config["model"]["masking"] is False

    assert config["loss"] == {
        "name": "sensitivity_specificity_weighted_cross_entropy",
        "alpha": 0.3,
        "beta": 0.5,
    }
    assert config["training"]["lr"] == pytest.approx(1e-4)
    assert config["training"]["weight_decay"] == pytest.approx(0.01)
    assert config["training"]["batch_size"] == 256
    assert config["training"]["max_epochs"] == 120
    assert config["training"]["early_stopping_patience"] == 15
    assert config["training"]["scheduler_patience"] == 5
    assert config["training"]["batch_size_cpu"] == 8
    assert config["training"]["lopo"]["val_fraction"] == pytest.approx(0.2)
    assert config["training"]["lopo"]["stratify"] is True
    assert config["training"]["lopo"]["split_unit"] == "window"
    assert config["postprocessing"]["smoothing"]["window"] == 3
    assert config["metrics"]["targets"]["fp_per_hour_max"] == 1.0


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


def test_load_edf_keeps_shape_and_duplicate_names(tmp_path: Path) -> None:
    """Файл уже на 256 Гц: форма [C, T] и исходные подписи, включая дубликат."""
    sfreq = 256
    duration_sec = 2
    n_samples = sfreq * duration_sec
    time = np.arange(n_samples) / sfreq
    signals = np.vstack(
        [
            80.0 * np.sin(2 * np.pi * 8 * time),
            40.0 * np.cos(2 * np.pi * 12 * time),
        ]
    )
    labels = ["T8-P8", "T8-P8"]
    path = tmp_path / "duplicate.edf"
    _write_edf(path, signals, labels, sfreq)

    recording = load_edf(path, target_sfreq=256)

    assert recording.data.shape == (2, n_samples)
    assert recording.data.dtype == np.float32
    assert recording.channel_names == ("T8-P8", "T8-P8")
    assert recording.sfreq == pytest.approx(256)
    assert recording.source_sfreq == pytest.approx(256)
    assert recording.duration_sec == pytest.approx(duration_sec)
    assert recording.data.shape[0] == len(recording.channel_names)


def test_load_edf_resamples_to_target_rate(tmp_path: Path) -> None:
    """128 Гц приводятся к 256 Гц, число отсчётов удваивается."""
    source_sfreq = 128
    duration_sec = 2
    n_samples = source_sfreq * duration_sec
    time = np.arange(n_samples) / source_sfreq
    signals = np.vstack([50.0 * np.sin(2 * np.pi * 5 * time)])
    path = tmp_path / "slow.edf"
    _write_edf(path, signals, ["FP1-F7"], source_sfreq)

    recording = load_edf(path, target_sfreq=256)

    assert recording.data.shape == (1, duration_sec * 256)
    assert recording.source_sfreq == pytest.approx(128)
    assert recording.sfreq == pytest.approx(256)
    assert recording.channel_names == ("FP1-F7",)


def test_load_edf_missing_and_corrupt(tmp_path: Path) -> None:
    missing = tmp_path / "absent.edf"
    with pytest.raises(FileNotFoundError, match="не найден"):
        load_edf(missing, target_sfreq=256)

    corrupt = tmp_path / "corrupt.edf"
    corrupt.write_bytes(b"this is not an edf file")
    with pytest.raises(EdfLoadError):
        load_edf(corrupt, target_sfreq=256)

    with pytest.raises(ValueError, match="положительным"):
        load_edf(corrupt, target_sfreq=0)


def test_check_patient_script_reports_summary_without_edf(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Сквозной запуск проверки chb01, когда на диске есть только сводка."""
    import importlib.util

    patient_dir = tmp_path / "chb01"
    patient_dir.mkdir()
    (patient_dir / "chb01-summary.txt").write_text(CHB01_EXCERPT, encoding="utf-8")

    spec = importlib.util.spec_from_file_location("check_patient", ROOT / "scripts" / "check_patient.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    exit_code = module.main(["--patient", "chb01", "--data-root", str(tmp_path)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Приступов всего: 1" in captured.out
    assert "2996–3036" in captured.out
    assert "EDF не найден" in captured.out


@pytest.mark.skipif(
    not os.environ.get("CHBMIT_SUMMARY_DIR"),
    reason="Каталог официальных сводок не задан переменной CHBMIT_SUMMARY_DIR.",
)
def test_official_physionet_summaries() -> None:
    """Полные сводки PhysioNet: интервалы, смена монтажа и файлы без часов."""
    directory = Path(os.environ["CHBMIT_SUMMARY_DIR"])

    chb01 = parse_summary(directory / "chb01-summary.txt")
    assert chb01.n_seizures == 7
    assert chb01.sampling_rate_hz == 256
    found = {
        record.file_name: [(item.start_sec, item.end_sec) for item in record.seizures]
        for record in chb01.seizure_records()
    }
    assert found == {
        "chb01_03.edf": [(2996, 3036)],
        "chb01_04.edf": [(1467, 1494)],
        "chb01_15.edf": [(1732, 1772)],
        "chb01_16.edf": [(1015, 1066)],
        "chb01_18.edf": [(1720, 1810)],
        "chb01_21.edf": [(327, 420)],
        "chb01_26.edf": [(1862, 1963)],
    }
    assert len(chb01.files[0].channels) == 23
    assert chb01.files[0].channels[14] == "T8-P8"
    assert chb01.files[0].channels[22] == "T8-P8"

    chb04 = parse_summary(directory / "chb04-summary.txt")
    by_name = {record.file_name: record for record in chb04.files}
    assert by_name["chb04_05.edf"].seizures[0].start_sec == 7804
    assert by_name["chb04_05.edf"].seizures[0].end_sec == 7853
    assert "ECG" not in by_name["chb04_05.edf"].channels
    assert by_name["chb04_07.edf"].channels[-1] == "ECG"
    assert [(item.start_sec, item.end_sec) for item in by_name["chb04_28.edf"].seizures] == [
        (1679, 1781),
        (3782, 3898),
    ]

    chb12 = parse_summary(directory / "chb12-summary.txt")
    chb12_files = {record.file_name: record for record in chb12.files}
    seizure_file = chb12_files["chb12_27.edf"]
    assert seizure_file.n_seizures == 6
    assert seizure_file.channels[0] == "F7-CS2"
    assert (seizure_file.seizures[4].start_sec, seizure_file.seizures[4].end_sec) == (2388, 2440)
    assert chb12_files["chb12_28.edf"].channels[0] == "F7"

    chb17 = parse_summary(directory / "chb17-summary.txt")
    assert chb17.patient_id == "chb17"
    assert chb17.files[0].file_name.startswith("chb17a_")
    assert "-" in chb17.files[0].channels

    chb24 = parse_summary(directory / "chb24-summary.txt")
    first = chb24.files[0]
    assert first.start_time is None
    assert [(item.start_sec, item.end_sec) for item in first.seizures] == [(480, 505), (2451, 2476)]
