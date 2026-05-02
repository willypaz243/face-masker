"""Operaciones de manipulación de imágenes."""

from __future__ import annotations

import cv2
import numpy as np

from .types import FaceBox


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
