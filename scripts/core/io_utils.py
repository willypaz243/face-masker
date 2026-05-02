"""Utilidades de entrada/salida (directorios y archivos)."""

from __future__ import annotations

import json
import os
from datetime import datetime

import cv2
import numpy as np

from .types import DetectionReport


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
