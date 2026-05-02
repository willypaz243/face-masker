#!/usr/bin/env python3
"""Herramienta CLI de detección y enmascaramiento facial local."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import cv2
import numpy as np

CASCADE_PATH: str = os.path.join(
    cv2.data.haarcascades,  # pyright: ignore[reportAttributeAccessIssue]
    "haarcascade_frontalface_default.xml",
)


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


def generate_output_dir(base_dir: str, prefix: str = "face_detection") -> str:
    """Genera un directorio de salida con timestamp para historial de detecciones.

    Args:
        base_dir: Directorio base donde crear la carpeta.
        prefix: Prefijo del nombre de carpeta.

    Returns:
        Ruta completa al directorio generado.
    """
    timestamp: str = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path: str = os.path.join(base_dir, f"{prefix}_{timestamp}")

    return output_path


def load_cascade_classifier() -> Any:
    """Carga el clasificador Haar Cascade desde OpenCV.

    Returns:
        Un objeto CascadeClassifier de OpenCV.

    Raises:
        FileNotFoundError: Si el archivo del cascada no existe.
    """
    if not os.path.exists(CASCADE_PATH):
        raise FileNotFoundError(f"Archivo del clasificador no encontrado: {CASCADE_PATH}")

    classifier: Any = cv2.CascadeClassifier(CASCADE_PATH)
    if classifier.empty():
        raise RuntimeError("No se pudo cargar el clasificador Haar Cascade.")

    return classifier


def detect_faces(gray_image: np.ndarray, classifier: Any) -> list[FaceBox]:
    """Ejecuta la detección de caras en una imagen en escala de grises.

    Args:
        gray_image: Imagen en escala de grises (numpy array).
        classifier: Clasificador Haar Cascade cargado.

    Returns:
        Lista de FaceBox con las coordenadas de cada cara detectada.
    """
    faces: list[FaceBox] = []

    raw_result: Any = classifier.detectMultiScale(
        gray_image,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(30, 30),
    )

    detected: np.ndarray = np.asarray(raw_result)

    if detected.size == 0:
        return faces

    for coords in detected:
        x, y, w, h = coords
        faces.append(FaceBox(x=int(x), y=int(y), width=int(w), height=int(h)))

    return faces


def draw_result_image(original_image: np.ndarray, face_boxes: list[FaceBox]) -> np.ndarray:
    """Dibuja cajas verdes sobre las caras detectadas.

    Args:
        original_image: Imagen original BGR.
        face_boxes: Lista de cajas delimitadoras.

    Returns:
        Imagen con cajas dibujadas.
    """
    result: np.ndarray = original_image.copy()

    for box in face_boxes:
        cv2.rectangle(
            result,
            (box.x, box.y),
            (box.x + box.width, box.y + box.height),
            (0, 255, 0),
            2,
        )

    return result


def create_mask_image(original_image: np.ndarray, face_boxes: list[FaceBox]) -> np.ndarray:
    """Crea una máscara negra con zonas blancas sobre las caras detectadas.

    Args:
        original_image: Imagen original (se usa solo para obtener el tamaño).
        face_boxes: Lista de cajas delimitadoras.

    Returns:
        Imagen de máscara (BGR, fondo negro, caras en blanco).
    """
    height: int = original_image.shape[0]
    width: int = original_image.shape[1]

    mask: np.ndarray = np.zeros((height, width, 3), dtype=np.uint8)

    for box in face_boxes:
        cv2.rectangle(
            mask,
            (box.x, box.y),
            (box.x + box.width, box.y + box.height),
            (255, 255, 255),
            -1,
        )

    return mask


def calculate_coverage(face_boxes: list[FaceBox], img_height: int, img_width: int) -> float:
    """Calcula el porcentaje de cobertura de la máscara.

    Args:
        face_boxes: Lista de cajas delimitadoras.
        img_height: Altura de la imagen original.
        img_width: Ancho de la imagen original.

    Returns:
        Porcentaje de cobertura (0.0 - 100.0).
    """
    total_area: int = img_height * img_width
    if total_area == 0:
        return 0.0

    face_area: int = sum(box.width * box.height for box in face_boxes)
    coverage: float = (face_area / total_area) * 100.0

    return round(coverage, 2)


def write_outputs(
    result_image: np.ndarray,
    mask_image: np.ndarray,
    report: DetectionReport,
    output_dir: str,
) -> None:
    """Guarda las imágenes de resultado, la máscara y el reporte JSON.

    Args:
        result_image: Imagen con cajas dibujadas.
        mask_image: Imagen de máscara.
        report: Reporte de detección.
        output_dir: Directorio donde guardar los archivos.
    """
    os.makedirs(output_dir, exist_ok=True)

    faces_path: str = os.path.join(output_dir, "detected_faces.jpg")
    mask_path: str = os.path.join(output_dir, "mask.png")

    cv2.imwrite(faces_path, result_image)
    cv2.imwrite(mask_path, mask_image)

    report_path: str = os.path.join(output_dir, "report.json")
    with open(report_path, "w") as file:
        json.dump(report.to_dict(), file, indent=2)


def build_report(
    face_boxes: list[FaceBox],
    image_shape: tuple[int, int, int],
) -> DetectionReport:
    """Construye el reporte de detección.

    Args:
        face_boxes: Lista de cajas delimitadoras detectadas.
        image_shape: Forma de la imagen (height, width, channels).

    Returns:
        Un objeto DetectionReport con todas las métricas.
    """
    height: int = image_shape[0]
    width: int = image_shape[1]

    report: DetectionReport = DetectionReport(
        face_detected=len(face_boxes) > 0,
        num_faces=len(face_boxes),
        bounding_boxes=face_boxes,
        image_size=(width, height),
        mask_coverage_pct=calculate_coverage(face_boxes, height, width),
    )

    if len(face_boxes) == 0:
        report.warnings.append("no se detectaron caras en la imagen")

    return report


def process_image(input_path: str, output_dir: str) -> DetectionReport:
    """Procesa una imagen: detecta caras, genera máscara y reporte.

    Args:
        input_path: Ruta a la imagen de entrada.
        output_dir: Directorio donde guardar los outputs.

    Returns:
        El reporte de detección generado.

    Raises:
        FileNotFoundError: Si la imagen de entrada no existe.
        ValueError: Si la imagen no se puede leer.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Imagen no encontrada: {input_path}")

    original_image: np.ndarray | None = cv2.imread(input_path)
    if original_image is None:
        raise ValueError(f"No se pudo leer la imagen: {input_path}")

    classifier = load_cascade_classifier()
    gray_image: np.ndarray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    face_boxes: list[FaceBox] = detect_faces(gray_image, classifier)

    report: DetectionReport = build_report(face_boxes, original_image.shape)

    result_image: np.ndarray = draw_result_image(original_image, face_boxes)
    mask_image: np.ndarray = create_mask_image(original_image, face_boxes)

    write_outputs(result_image, mask_image, report, output_dir)

    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parsea los argumentos de la línea de comandos.

    Args:
        argv: Lista de argumentos (por defecto sys.argv[1:]).

    Returns:
        Namespace con los argumentos parseados.
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Detección y enmascaramiento facial local"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Ruta a la imagen de entrada",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directorio base de salida (por defecto: outputs/)",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Punto de entrada principal.

    Args:
        argv: Lista de argumentos (para testing).

    Returns:
        Código de salida (0 = éxito, 1 = error).
    """
    args = parse_args(argv)

    final_output_dir: str = generate_output_dir(args.output_dir)

    try:
        report = process_image(args.input, final_output_dir)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    msg: str = f"Caras encontradas: {report.num_faces} -> {final_output_dir}/"
    print(msg)

    if len(report.warnings) > 0:
        for warning in report.warnings:
            print(f"  Aviso: {warning}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
