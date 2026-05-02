#!/usr/bin/env python3
"""Generador de imágenes sintéticas para pruebas."""

from __future__ import annotations

import os

import cv2
import numpy as np


def generate_test_image(
    output_path: str = "test_synthetic.jpg",
    img_width: int = 500,
    img_height: int = 500,
) -> np.ndarray:
    """Genera una imagen sintética con círculos de colores.

    Esta imagen no contendrá patrones faciales reales,
    por lo que Haar Cascade detectará 0 caras.
    Es útil para probar el manejo de casos sin detecciones.

    Args:
        output_path: Ruta donde guardar la imagen generada.
        img_width: Ancho de la imagen en píxeles.
        img_height: Alto de la imagen en píxeles.

    Returns:
        La imagen generada como array numpy.
    """
    image: np.ndarray = np.zeros((img_height, img_width, 3), dtype=np.uint8)

    # Fondo azul oscuro
    cv2.rectangle(image, (0, 0), (img_width, img_height), (50, 50, 150), -1)

    # Círculos decorativos de colores
    colors: list[tuple[int, int, int]] = [
        (255, 100, 100),
        (100, 255, 100),
        (100, 100, 255),
        (255, 255, 100),
        (255, 100, 255),
    ]

    centers: list[tuple[int, int]] = [
        (125, 125),
        (375, 125),
        (125, 375),
        (375, 375),
        (250, 250),
    ]

    for color, center in zip(colors, centers, strict=True):
        cv2.circle(image, center, 60, color, -1)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cv2.imwrite(output_path, image)

    return image


def main() -> None:
    """Ejecuta la generación de imagen sintética."""
    output: str = "test_synthetic.jpg"
    image = generate_test_image(output)
    print(f"Imagen sintética generada: {output} ({image.shape[1]}x{image.shape[0]})")


if __name__ == "__main__":
    main()
