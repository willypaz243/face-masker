"""Pipeline de procesamiento de imágenes."""

from __future__ import annotations

import os

import cv2
import numpy as np

from .detector import FaceDetector
from .image_ops import create_mask_image, draw_result_image
from .io_utils import write_outputs
from .metrics import build_report
from .types import DetectionReport, FaceBox


def process_image(
    input_path: str,
    output_dir: str,
    detector: FaceDetector,
) -> DetectionReport:
    """Procesa una imagen: detecta caras con el detector dado, genera máscara y reporte.

    Args:
        input_path: Ruta a la imagen de entrada.
        output_dir: Directorio donde guardar los outputs.
        detector: Detector de caras a utilizar.

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

    face_boxes: list[FaceBox] = detector.detect(original_image)

    report: DetectionReport = build_report(face_boxes, original_image.shape)

    result_image: np.ndarray = draw_result_image(original_image, face_boxes)
    mask_image: np.ndarray = create_mask_image(original_image, face_boxes)

    write_outputs(result_image, mask_image, report, output_dir)

    return report
