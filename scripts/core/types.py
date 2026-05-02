"""Tipos de datos compartidos por todos los módulos."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class DetectionStrategy(str, Enum):
    """Estrategias de detección facial soportadas.

    Cada valor corresponde al nombre que se usa con el argumento --strategy.
    """

    HAAR = "haar"
    YUNET = "yunet"
    DLIB = "dlib"


@dataclass(frozen=True)
class FaceBox:
    """Caja delimitadora de una cara detectada."""

    x: int
    y: int
    width: int
    height: int


@dataclass
class DetectionReport:
    """Reporte JSON de la detección facial."""

    face_detected: bool
    num_faces: int
    bounding_boxes: list[FaceBox] = field(default_factory=list)
    image_size: tuple[int, int] = (0, 0)
    mask_coverage_pct: float = 0.0
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convierte el reporte a un diccionario serializable en JSON."""
        return {
            "face_detected": self.face_detected,
            "num_faces": self.num_faces,
            "bounding_boxes": [
                {"x": fb.x, "y": fb.y, "w": fb.width, "h": fb.height} for fb in self.bounding_boxes
            ],
            "image_size": list(self.image_size),
            "mask_coverage_pct": self.mask_coverage_pct,
            "warnings": self.warnings,
        }
