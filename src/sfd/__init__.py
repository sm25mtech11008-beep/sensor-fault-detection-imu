"""Sensor fault detection for automotive IMU / thermometer channels."""
from .config import CLASS_LABELS, DATASET_META, Config
from .data import build_windows, signal_statistics
from .faults import FaultConfig, inject_dataset
from .features import FEATURE_NAMES, extract
from .pipeline import Dataset, build_dataset
from .splits import leakage_report, make_split

__version__ = "2.0.0"
__all__ = [
    "Config", "CLASS_LABELS", "DATASET_META", "build_windows", "signal_statistics",
    "FaultConfig", "inject_dataset", "FEATURE_NAMES", "extract",
    "Dataset", "build_dataset", "make_split", "leakage_report",
]
