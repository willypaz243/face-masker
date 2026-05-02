"""Pruebas automatizadas para la herramienta de detección facial."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

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
    generate_output_dir,
    process_image,
)
from scripts.core.detector import HaarDetector, get_detector
from scripts.detect_faces_and_mask import parse_args

TEST_IMAGES_DIR: str = os.path.join(os.path.dirname(__file__), "..", "test_images")


def _get_test_image_path(filename: str) -> str:
    """Retorna la ruta completa a una imagen de prueba."""
    return os.path.join(TEST_IMAGES_DIR, filename)


# --- Helpers ---

HAAR_DETECTOR: HaarDetector = HaarDetector()


# --- Pruebas de funciones puras ---


def test_load_cascade_classifier_returns_valid_classifer() -> None:
    """Verifica que el clasificador se carga correctamente."""
    detector = get_detector("haar")
    assert hasattr(detector, "name")
    assert detector.name == "haar"


def test_detect_faces_returns_empty_list_on_blank_image() -> None:
    """Una imagen en blanco no debe detectar caras."""
    blank: np.ndarray = np.zeros((100, 100), dtype=np.uint8)

    faces = HAAR_DETECTOR.detect(blank)

    assert isinstance(faces, list)
    assert len(faces) == 0


def test_detect_faces_returns_list_of_facebox() -> None:
    """detect debe retornar una lista de FaceBox."""
    blank: np.ndarray = np.zeros((100, 100), dtype=np.uint8)

    faces = HAAR_DETECTOR.detect(blank)

    for face in faces:
        assert isinstance(face, FaceBox)


def test_draw_result_image_returns_same_shape() -> None:
    """draw_result_image debe retornar imagen del mismo tamaño."""
    image: np.ndarray = np.zeros((200, 200, 3), dtype=np.uint8)
    boxes: list[FaceBox] = [FaceBox(x=50, y=50, width=30, height=30)]

    result = draw_result_image(image, boxes)

    assert result.shape == image.shape


def test_create_mask_image_returns_correct_shape() -> None:
    """create_mask_image debe retornar imagen del tamaño correcto."""
    image: np.ndarray = np.zeros((200, 200, 3), dtype=np.uint8)
    boxes: list[FaceBox] = [FaceBox(x=50, y=50, width=30, height=30)]

    mask = create_mask_image(image, boxes)

    assert mask.shape == image.shape


def test_create_mask_image_has_white_pixels_when_faces_exist() -> None:
    """La máscara debe tener píxeles blancos cuando hay caras."""
    image: np.ndarray = np.zeros((200, 200, 3), dtype=np.uint8)
    boxes: list[FaceBox] = [FaceBox(x=50, y=50, width=30, height=30)]

    mask = create_mask_image(image, boxes)

    white_pixels: int = cv2.countNonZero(cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY))
    assert white_pixels > 0


def test_create_mask_image_is_all_black_when_no_faces() -> None:
    """La máscara debe ser toda negra cuando no hay caras."""
    image: np.ndarray = np.zeros((200, 200, 3), dtype=np.uint8)
    boxes: list[FaceBox] = []

    mask = create_mask_image(image, boxes)

    white_pixels: int = cv2.countNonZero(cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY))
    assert white_pixels == 0


def test_calculate_coverage_zero_when_no_faces() -> None:
    """La cobertura debe ser 0 cuando no hay caras."""
    coverage = calculate_coverage([], 100, 100)
    assert coverage == 0.0


def test_calculate_coverage_returns_positive_value() -> None:
    """La cobertura debe ser positiva cuando hay caras que ocupan área."""
    boxes: list[FaceBox] = [FaceBox(x=0, y=0, width=10, height=10)]
    coverage = calculate_coverage(boxes, 100, 100)
    assert coverage == pytest.approx(1.0)


def test_calculate_coverage_returns_zero_on_zero_area() -> None:
    """La cobertura debe ser 0 si la imagen tiene área 0."""
    boxes: list[FaceBox] = [FaceBox(x=0, y=0, width=10, height=10)]
    coverage = calculate_coverage(boxes, 0, 0)
    assert coverage == 0.0


def test_generate_output_dir_creates_path_with_timestamp() -> None:
    """generate_output_dir debe crear ruta con prefijo y timestamp."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = generate_output_dir(tmpdir, "test")

        assert os.path.isdir(tmpdir)
        assert os.path.basename(output_path).startswith("test_")
        assert len(os.path.basename(output_path)) > len("test_")


def test_generate_output_dir_uses_default_prefix() -> None:
    """generate_output_dir debe usar 'face_detection' como prefijo por defecto."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = generate_output_dir(tmpdir)

        assert os.path.basename(output_path).startswith("face_detection_")


# --- Pruebas de reportes ---


def test_build_report_no_faces() -> None:
    """Verifica el reporte cuando no se detectan caras."""
    boxes: list[FaceBox] = []

    report = build_report(boxes, (100, 100, 3))

    assert report.face_detected is False
    assert report.num_faces == 0
    assert len(report.bounding_boxes) == 0
    assert len(report.warnings) > 0
    assert report.mask_coverage_pct == 0.0


def test_build_report_with_faces() -> None:
    """Verifica el reporte cuando se detectan caras."""
    boxes: list[FaceBox] = [FaceBox(x=10, y=20, width=30, height=40)]

    report = build_report(boxes, (100, 100, 3))

    assert report.face_detected is True
    assert report.num_faces == 1
    assert len(report.bounding_boxes) == 1
    assert report.image_size == (100, 100)


def test_report_to_dict_has_required_fields() -> None:
    """Verifica que to_dict() incluye todos los campos requeridos."""
    boxes: list[FaceBox] = [FaceBox(x=10, y=20, width=30, height=40)]
    report = build_report(boxes, (100, 100, 3))

    d: dict = report.to_dict()

    required_keys: list[str] = [
        "face_detected",
        "num_faces",
        "bounding_boxes",
        "image_size",
        "mask_coverage_pct",
        "warnings",
    ]

    for key in required_keys:
        assert key in d, f"Falta el campo requerido: {key}"


# --- Pruebas de CLI ---


def test_parse_args_with_input() -> None:
    """Verifica que parse_args procesa --input correctamente."""
    args = parse_args(["--input", "test.jpg"])

    assert args.input == "test.jpg"


def test_parse_args_default_output_dir() -> None:
    """Verifica que el output-dir tiene valor por defecto 'outputs'."""
    args = parse_args(["--input", "test.jpg"])

    assert args.output_dir == "outputs"


def test_parse_args_custom_output_dir() -> None:
    """Verifica que se puede especificar un output-dir personalizado."""
    args = parse_args(["--input", "test.jpg", "--output-dir", "custom/out"])

    assert args.output_dir == "custom/out"


def test_parse_args_default_strategy_haar() -> None:
    """Verifica que --strategy default es 'haar'."""
    args = parse_args(["--input", "test.jpg"])

    assert args.strategy == "haar"


def test_parse_args_custom_strategy() -> None:
    """Verifica que se puede especificar una estrategia personalizada."""
    args = parse_args(["--input", "test.jpg", "--strategy", "yunet"])

    assert args.strategy == "yunet"


# --- Pruebas de integración con imágenes reales ---


def test_process_image_with_gates_linus() -> None:
    """Ejecuta el pipeline con la imagen de gates-linus.jpg."""
    img_path = _get_test_image_path("gates-linus.jpg")

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = os.path.join(tmpdir, "output")

        report = process_image(img_path, out_dir, HAAR_DETECTOR)

        assert isinstance(report, DetectionReport)
        assert os.path.exists(os.path.join(out_dir, "detected_faces.jpg"))
        assert os.path.exists(os.path.join(out_dir, "mask.png"))
        assert os.path.exists(os.path.join(out_dir, "report.json"))


def test_process_image_with_gemini_generated() -> None:
    """Ejecuta el pipeline con la imagen Gemini_Generated_Image.png."""
    img_path = _get_test_image_path("Gemini_Generated_Image.png")

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = os.path.join(tmpdir, "output")

        report = process_image(img_path, out_dir, HAAR_DETECTOR)

        assert isinstance(report, DetectionReport)
        assert os.path.exists(os.path.join(out_dir, "detected_faces.jpg"))
        assert os.path.exists(os.path.join(out_dir, "mask.png"))
        assert os.path.exists(os.path.join(out_dir, "report.json"))


def test_process_image_with_test_random_image() -> None:
    """Ejecuta el pipeline con la imagen test_random_image.jpg."""
    img_path = _get_test_image_path("test_random_image.jpg")

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = os.path.join(tmpdir, "output")

        report = process_image(img_path, out_dir, HAAR_DETECTOR)

        assert isinstance(report, DetectionReport)
        assert os.path.exists(os.path.join(out_dir, "detected_faces.jpg"))
        assert os.path.exists(os.path.join(out_dir, "mask.png"))
        assert os.path.exists(os.path.join(out_dir, "report.json"))


def test_process_image_creates_valid_json() -> None:
    """Verifica que el JSON generado tiene la estructura correcta."""
    img_path = _get_test_image_path("gates-linus.jpg")

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = os.path.join(tmpdir, "output")

        process_image(img_path, out_dir, HAAR_DETECTOR)

        json_path = os.path.join(out_dir, "report.json")

        with open(json_path) as f:
            data: dict = json.load(f)

        assert "face_detected" in data
        assert "num_faces" in data
        assert "bounding_boxes" in data
        assert "image_size" in data
        assert "mask_coverage_pct" in data
        assert "warnings" in data


def test_process_image_with_nonexistent_file() -> None:
    """Verifica que process_image levanta FileNotFoundError para archivo inexistente."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = os.path.join(tmpdir, "output")

        with pytest.raises(FileNotFoundError):
            process_image("/no/existe/imagen.jpg", out_dir, HAAR_DETECTOR)


def test_process_image_with_invalid_image_format() -> None:
    """Verifica que process_image levanta ValueError para formato inválido."""
    with tempfile.TemporaryDirectory() as tmpdir:
        invalid_path = os.path.join(tmpdir, "invalid.txt")

        with open(invalid_path, "w") as f:
            f.write("not an image")

        out_dir = os.path.join(tmpdir, "output")

        with pytest.raises(ValueError):
            process_image(invalid_path, out_dir, HAAR_DETECTOR)


def test_integration_with_real_haar_cascade() -> None:
    """Verifica que el clasificador funciona con una imagen de prueba."""
    img_path = _get_test_image_path("test_random_image.jpg")

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = os.path.join(tmpdir, "output")

        report = process_image(img_path, out_dir, HAAR_DETECTOR)

        assert isinstance(report, DetectionReport)
        assert os.path.exists(os.path.join(out_dir, "report.json"))


def test_main_returns_zero_on_success() -> None:
    """Verifica que main retorna 0 en ejecución exitosa."""
    from scripts.detect_faces_and_mask import main

    img_path = _get_test_image_path("gates-linus.jpg")

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = os.path.join(tmpdir, "output")

        exit_code = main(["--input", img_path, "--output-dir", out_dir])

        assert exit_code == 0


def test_main_returns_one_on_failure() -> None:
    """Verifica que main retorna 1 cuando hay error."""
    from scripts.detect_faces_and_mask import main

    exit_code = main(["--input", "/no/existe/img.jpg"])

    assert exit_code == 1


def test_main_creates_timestamped_output_folder() -> None:
    """Verifica que main crea carpetas con timestamp para historial."""
    from scripts.detect_faces_and_mask import main

    img_path = _get_test_image_path("gates-linus.jpg")

    with tempfile.TemporaryDirectory() as tmpdir:
        exit_code = main(["--input", img_path, "--output-dir", tmpdir])

        assert exit_code == 0

        subdirs = [d for d in os.listdir(tmpdir) if os.path.isdir(os.path.join(tmpdir, d))]

        assert len(subdirs) >= 1
        assert any(d.startswith("face_detection_") for d in subdirs)


# --- Pruebas de estrategia ---


def test_get_detector_haar_returns_valid_detector() -> None:
    """get_detector con 'haar' retorna un detector funcional."""
    detector = get_detector("haar")
    assert detector.name == "haar"


def test_get_detector_unknown_raises_value_error() -> None:
    """get_detector con estrategia inválida levanta ValueError."""
    with pytest.raises(ValueError, match="no soportada"):
        get_detector("invalid_strategy")


def test_detection_strategy_enum_values() -> None:
    """Valida los valores del enum DetectionStrategy."""
    from scripts.core.types import DetectionStrategy

    assert DetectionStrategy.HAAR.value == "haar"
    assert DetectionStrategy.YUNET.value == "yunet"
    assert DetectionStrategy.DLIB.value == "dlib"
