"""Single source of truth for paths, windowing and dataset metadata.

Replaces the ``BASE_PATH = r'C:\PVS_Dataset'`` line that was duplicated across
six files (five notebooks plus ``scripts/generate_report.py``). Override the data location without editing code:

    set PVS_DATA=D:\somewhere\else      (Windows)
    export PVS_DATA=/data/pvs             (Linux/macOS)
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# ── Dataset metadata ────────────────────────────────────────────────────
# 9 drives = 3 vehicles x 3 drivers x 3 road scenarios.
# `vehicle` is the grouping key that makes leave-one-vehicle-out possible.
DATASET_META: dict[str, dict[str, str]] = {
    "PVS1": {"vehicle": "VW Saveiro", "driver": "Driver 1", "scenario": "Scenario 1"},
    "PVS2": {"vehicle": "VW Saveiro", "driver": "Driver 1", "scenario": "Scenario 2"},
    "PVS3": {"vehicle": "VW Saveiro", "driver": "Driver 1", "scenario": "Scenario 3"},
    "PVS4": {"vehicle": "Fiat Bravo", "driver": "Driver 2", "scenario": "Scenario 1"},
    "PVS5": {"vehicle": "Fiat Bravo", "driver": "Driver 2", "scenario": "Scenario 2"},
    "PVS6": {"vehicle": "Fiat Bravo", "driver": "Driver 2", "scenario": "Scenario 3"},
    "PVS7": {"vehicle": "Fiat Palio", "driver": "Driver 3", "scenario": "Scenario 1"},
    "PVS8": {"vehicle": "Fiat Palio", "driver": "Driver 3", "scenario": "Scenario 2"},
    "PVS9": {"vehicle": "Fiat Palio", "driver": "Driver 3", "scenario": "Scenario 3"},
}

SIDES = ("left", "right")

# The MPU-9250 exposes three thermometers, one per mounting point. The
# original pipeline silently used only the first match for 'temp'.
TEMP_CHANNELS = ("temp_dashboard", "temp_above_suspension", "temp_below_suspension")

CLASS_LABELS: dict[int, str] = {
    0: "normal",
    1: "noise_fault",
    2: "bias_fault",
    3: "flatline_fault",
    4: "stuck_fault",
    5: "dropout_fault",
    6: "scale_fault",
}

CLASS_COLORS: dict[str, str] = {
    "normal": "#43A047",
    "noise_fault": "#E53935",
    "bias_fault": "#FB8C00",
    "flatline_fault": "#8E24AA",
    "stuck_fault": "#00897B",
    "dropout_fault": "#5D4037",
    "scale_fault": "#C2185B",
}


def _default_data_root() -> Path:
    env = os.environ.get("PVS_DATA")
    if env:
        return Path(env)
    return Path("C:/PVS_Dataset") if os.name == "nt" else Path.home() / "PVS_Dataset"


@dataclass(frozen=True)
class Config:
    """Everything a run needs. Immutable so experiments cannot mutate shared state."""

    data_root: Path = field(default_factory=_default_data_root)
    results_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2] / "results")

    # Windowing
    window_size: int = 100          # 1.0 s at 100 Hz
    window_step: int = 50           # 50 % overlap (kept for parity with v1)
    sampling_hz: int = 100

    # Which temperature channel to model. None -> all three.
    channel: str | None = "temp_above_suspension"

    seed: int = 42

    # Class set actually used (subset of CLASS_LABELS)
    n_classes: int = 4

    def __post_init__(self) -> None:
        object.__setattr__(self, "data_root", Path(self.data_root))
        object.__setattr__(self, "results_root", Path(self.results_root))

    @property
    def class_names(self) -> list[str]:
        return [CLASS_LABELS[i] for i in range(self.n_classes)]

    def drive_path(self, dataset_id: str, side: str) -> Path:
        return self.data_root / dataset_id / f"dataset_mpu_{side}.csv"

    def validate(self) -> None:
        """Fail loudly and early if the dataset is missing or incomplete."""
        missing = [
            str(self.drive_path(d, s))
            for d in DATASET_META
            for s in SIDES
            if not self.drive_path(d, s).exists()
        ]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} PVS file(s) not found under {self.data_root}.\n"
                f"First missing: {missing[0]}\n"
                "Run `python -m sfd.cli download` or set PVS_DATA to the right folder."
            )
