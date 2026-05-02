"""Re-exporta los módulos core para importaciones limpias."""

from .detector import DlibDetector, FaceDetector, HaarDetector, YuNetDetector, get_detector
from .image_ops import create_mask_image, draw_result_image
from .io_utils import generate_output_dir, write_outputs
from .metrics import build_report, calculate_coverage
from .pipeline import process_image
from .types import DetectionReport, DetectionStrategy, FaceBox

__all__: list[str] = [
    "FaceDetector",
    "HaarDetector",
    "YuNetDetector",
    "DlibDetector",
    "get_detector",
    "DetectionStrategy",
    "DetectionReport",
    "FaceBox",
    "draw_result_image",
    "create_mask_image",
    "calculate_coverage",
    "build_report",
    "generate_output_dir",
    "write_outputs",
    "process_image",
]
