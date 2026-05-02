"""Cálculo de métricas y construcción de reportes."""

from __future__ import annotations

from .types import DetectionReport, FaceBox


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
