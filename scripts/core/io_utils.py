"""Utilidades de entrada/salida (directorios y archivos)."""

from __future__ import annotations

import json
import os
from datetime import datetime

import cv2
import numpy as np

from .types import DetectionReport

OUTPUT_BASE: str = "outputs"


def resolve_output_dir(user_path: str) -> str:
    """Valida y resuelve el directorio de salida.

    - Debe comenzar con 'outputs/'
    - Si solo se pasa 'outputs', genera un subdirectorio con timestamp
    - Crea 'outputs/' si no existe

    Args:
        user_path: Ruta proporcionada por el usuario vía --output-dir.

    Returns:
        Ruta absoluta resuelta al directorio de salida final.

    Raises:
        ValueError: Si la ruta no comienza con 'outputs/'.
    """
    abs_user: str = os.path.abspath(user_path)
    abs_base: str = os.path.abspath(OUTPUT_BASE)

    if not (abs_user == abs_base or abs_user.startswith(abs_base + os.sep)):
        raise ValueError(
            f"El directorio de salida debe comenzar con '{OUTPUT_BASE}/', recibido: '{user_path}'"
        )

    os.makedirs(abs_base, exist_ok=True)

    if abs_user == abs_base:
        timestamp: str = datetime.now().strftime("%Y%m%d_%H%M%S")
        resolved: str = os.path.join(abs_base, timestamp)
    else:
        resolved = abs_user

    os.makedirs(resolved, exist_ok=True)
    return resolved


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
