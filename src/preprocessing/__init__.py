"""Предварительная обработка многоканальной ЭЭГ (EEG).

Сейчас доступны чтение сводок приступов и загрузка EDF.
Фильтрация, сопоставление каналов, сегментация и нормализация
подключаются следующими шагами фазы 1.
"""

from src.preprocessing.annotations import (
    AnnotationParseError,
    EdfRecord,
    PatientSummary,
    SeizureInterval,
    parse_summary,
    parse_summary_text,
)
from src.preprocessing.edf_loader import EdfLoadError, EdfRecording, load_edf

__all__ = [
    "AnnotationParseError",
    "EdfLoadError",
    "EdfRecord",
    "EdfRecording",
    "PatientSummary",
    "SeizureInterval",
    "load_edf",
    "parse_summary",
    "parse_summary_text",
]
