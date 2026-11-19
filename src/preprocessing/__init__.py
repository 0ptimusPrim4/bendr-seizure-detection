"""Предварительная обработка многоканальной ЭЭГ (EEG)."""

from src.preprocessing.annotations import (
    AnnotationParseError,
    EdfRecord,
    PatientSummary,
    SeizureInterval,
    parse_summary,
    parse_summary_text,
)
from src.preprocessing.channel_mapping import ChannelMappingError, map_channels
from src.preprocessing.edf_loader import EdfLoadError, EdfRecording, load_edf
from src.preprocessing.filtering import FilterError, apply_bandpass_filter
from src.preprocessing.normalization import EEGNormalizer, NormalizerError
from src.preprocessing.segmentation import SegmentationError, segment_and_label

__all__ = [
    "AnnotationParseError",
    "ChannelMappingError",
    "EEGNormalizer",
    "EdfLoadError",
    "EdfRecord",
    "EdfRecording",
    "FilterError",
    "NormalizerError",
    "PatientSummary",
    "SegmentationError",
    "SeizureInterval",
    "apply_bandpass_filter",
    "load_edf",
    "map_channels",
    "parse_summary",
    "parse_summary_text",
    "segment_and_label",
]
