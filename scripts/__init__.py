"""Compatibilidad con tests — re-exporta todos los nombres públicos."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from scripts.core import (
    DetectionReport,
    FaceBox,
    calculate_coverage,
    create_mask_image,
    draw_result_image,
    generate_output_dir,
)
from scripts.core.detector import HaarDetector
from scripts.detect_faces_and_mask import main, parse_args


def load_cascade_classifier() -> HaarDetector:
    """Legacy wrapper para compatibilidad con tests existentes."""
    return HaarDetector()


def detect_faces(gray: np.ndarray, classifier: HaarDetector) -> Sequence[FaceBox]:
    """Legacy wrapper para compatibilidad con tests existentes."""
    return classifier.detect(gray)  # type: ignore[return-value]


__all__: list[str] = [
    "FaceBox",
    "DetectionReport",
    "calculate_coverage",
    "create_mask_image",
    "draw_result_image",
    "generate_output_dir",
    "load_cascade_classifier",
    "detect_faces",
    "parse_args",
    "main",
]
