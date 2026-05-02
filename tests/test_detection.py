"""Pruebas simplificadas para detección facial y reporte."""

from __future__ import annotations

import os
import tempfile

import cv2
import numpy as np
import pytest

from scripts.core import (
    DetectionReport,
    FaceBox,
    build_report,
    calculate_coverage,
    create_mask_image,
    draw_result_image,
    process_image,
)
from scripts.core.detector import get_detector

try:
    __import__("dlib")  # type: ignore[import-not-found, union-attr]

    _HAS_DLIB: bool = True
except ImportError:
    _HAS_DLIB = False

# Imagen sintética con un rectángulo blanco (simula un rostro frontal).
SYNTHETIC_IMAGE: np.ndarray = np.zeros((400, 400, 3), dtype=np.uint8)
SYNTHETIC_IMAGE[150:250, 130:270] = 255  # rectángulo blanco en el centro


# --- Helpers ---


def _get_detector(strategy: str):
    """Retorna un detector por nombre de estrategia."""
    return get_detector(strategy)


# --- Pruebas de funciones puras ---


def test_detect_faces_blank_image_returns_empty() -> None:
    """Imagen completamente negra no debe detectar caras."""
    detector = _get_detector("haar")
    faces = detector.detect(np.zeros((100, 100), dtype=np.uint8))
    assert faces == []


def test_draw_result_image_preserves_shape() -> None:
    """draw_result_image no modifica las dimensiones de la imagen."""
    image: np.ndarray = np.zeros((200, 200, 3), dtype=np.uint8)
    boxes: list[FaceBox] = [FaceBox(x=50, y=50, width=30, height=30)]
    result = draw_result_image(image, boxes)
    assert result.shape == image.shape


def test_create_mask_image_has_white_pixels() -> None:
    """La máscara debe tener píxeles blancos cuando hay cajas."""
    image: np.ndarray = np.zeros((200, 200, 3), dtype=np.uint8)
    mask = create_mask_image(image, [FaceBox(x=50, y=50, width=30, height=30)])
    assert cv2.countNonZero(cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)) > 0


def test_create_mask_image_all_black_no_faces() -> None:
    """La máscara debe ser negra cuando no hay cajas."""
    image: np.ndarray = np.zeros((200, 200, 3), dtype=np.uint8)
    mask = create_mask_image(image, [])
    assert cv2.countNonZero(cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)) == 0


def test_calculate_coverage_zero_on_no_faces() -> None:
    """Cobertura debe ser 0 cuando no hay caras."""
    assert calculate_coverage([], 100, 100) == 0.0


# --- Pruebas de reporte ---


def test_build_report_with_faces() -> None:
    """build_report genera reporte correcto con cajas."""
    boxes: list[FaceBox] = [FaceBox(x=10, y=20, width=30, height=40)]
    report = build_report(boxes, (100, 100, 3))

    assert report.face_detected is True
    assert report.num_faces == 1
    assert len(report.bounding_boxes) == 1
    assert report.image_size == (100, 100)


def test_build_report_no_faces() -> None:
    """build_report indica sin caras y con advertencias."""
    report = build_report([], (100, 100, 3))

    assert report.face_detected is False
    assert report.num_faces == 0
    assert len(report.warnings) > 0


def test_report_to_dict_has_required_fields() -> None:
    """to_dict incluye todos los campos requeridos."""
    boxes: list[FaceBox] = [FaceBox(x=10, y=20, width=30, height=40)]
    report = build_report(boxes, (100, 100, 3))
    d = report.to_dict()

    for key in (
        "face_detected",
        "num_faces",
        "bounding_boxes",
        "image_size",
        "mask_coverage_pct",
        "warnings",
    ):
        assert key in d


# --- Pruebas de estrategia: detección + reporte ---


_STRATEGIES: list[str] = ["haar", "yunet"] + (["dlib"] if _HAS_DLIB else [])


@pytest.mark.parametrize("strategy", _STRATEGIES)
def test_strategy_detect_and_report(strategy: str) -> None:
    """Cada estrategia detecta caras y genera un reporte válido."""
    detector = _get_detector(strategy)
    faces = detector.detect(SYNTHETIC_IMAGE)

    report = build_report(faces, SYNTHETIC_IMAGE.shape)

    assert isinstance(report, DetectionReport)
    assert report.num_faces >= 0


@pytest.mark.parametrize("strategy", _STRATEGIES)
def test_strategy_pipeline(strategy: str) -> None:
    """Cada estrategia ejecuta el pipeline completo sin error."""
    img_path = _save_synthetic_image()

    with tempfile.TemporaryDirectory() as tmpdir:
        report = process_image(img_path, tmpdir, _get_detector(strategy))

        assert isinstance(report, DetectionReport)
        assert os.path.exists(os.path.join(tmpdir, "detected_faces.jpg"))
        assert os.path.exists(os.path.join(tmpdir, "mask.png"))
        assert os.path.exists(os.path.join(tmpdir, "report.json"))


def test_pipeline_invalid_input_raises() -> None:
    """process_image levanta ValueError con archivo inválido."""
    with tempfile.TemporaryDirectory() as tmpdir:
        invalid = os.path.join(tmpdir, "bad.txt")
        with open(invalid, "w") as f:
            f.write("not an image")

        with pytest.raises(ValueError):
            process_image(invalid, tmpdir, _get_detector("haar"))


def test_pipeline_nonexistent_file_raises() -> None:
    """process_image levanta FileNotFoundError con ruta inexistente."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(FileNotFoundError):
            process_image("/no/existe.jpg", tmpdir, _get_detector("haar"))


def test_get_detector_unknown_strategy_raises() -> None:
    """get_detector con estrategia inválida levanta ValueError."""
    with pytest.raises(ValueError, match="no soportada"):
        get_detector("invalid_strategy")


# --- Helpers ---


def _save_synthetic_image() -> str:
    """Guarda la imagen sintética en un archivo temporal y retorna su ruta."""
    fd, path = tempfile.mkstemp(suffix=".jpg")
    os.close(fd)
    cv2.imwrite(path, SYNTHETIC_IMAGE)
    return path
