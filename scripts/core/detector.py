"""Base abstracta para detectores y registry de estrategias."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from .types import DetectionStrategy, FaceBox


class FaceDetector:
    """Clase base abstracta para detectores de caras."""

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def detect(self, image: np.ndarray) -> list[FaceBox]:
        raise NotImplementedError


class HaarDetector(FaceDetector):
    """Detector basado en Haar Cascade embebido en OpenCV."""

    _CASCADE_PATH: str = cv2.data.haarcascades + "/haarcascade_frontalface_default.xml"  # pyright: ignore[reportAttributeAccessIssue]

    def __init__(self) -> None:
        self._classifier = cv2.CascadeClassifier(self._CASCADE_PATH)
        if self._classifier.empty():
            raise RuntimeError("No se pudo cargar el clasificador Haar Cascade.")

    @property
    def name(self) -> str:
        return DetectionStrategy.HAAR.value

    def detect(self, image: np.ndarray) -> list[FaceBox]:
        """Ejecuta la detección de caras en una imagen (BGR o gris).

        Args:
            image: Imagen original BGR o escala de grises.

        Returns:
            Lista de FaceBox con las coordenadas de cada cara detectada.
        """
        gray: np.ndarray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        raw_result = self._classifier.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(30, 30),
        )

        detected: np.ndarray = np.asarray(raw_result)

        if detected.size == 0:
            return []

        return [
            FaceBox(x=int(coords[0]), y=int(coords[1]), width=int(coords[2]), height=int(coords[3]))
            for coords in detected
        ]


class YuNetDetector(FaceDetector):
    """Detector basado en YuNet (ONNX) de OpenCV Zoo.

    Args:
        model_path: Ruta al archivo .onnx del modelo YuNet.
        input_size: Tamaño de entrada (width, height). Default (320, 320).
        conf_threshold: Umbral de confianza para detecciones. Default 0.6.
        nms_threshold: Umbral de NMS para supresión de cajas. Default 0.3.
    """

    def __init__(
        self,
        model_path: str = "",
        input_size: tuple[int, int] = (320, 320),
        conf_threshold: float = 0.6,
        nms_threshold: float = 0.3,
    ) -> None:
        if not model_path:
            raise ValueError("model_path es requerido para YuNetDetector")

        self._model_path = model_path
        self._input_size = input_size
        self._conf_threshold = conf_threshold
        self._nms_threshold = nms_threshold
        self._detector = cv2.FaceDetectorYN.create(
            model=model_path,
            config="",
            input_size=self._input_size,
            score_threshold=self._conf_threshold,
            nms_threshold=self._nms_threshold,
            top_k=5000,
            backend_id=0,
            target_id=0,
        )

    @property
    def name(self) -> str:
        return DetectionStrategy.YUNET.value

    def detect(self, image: np.ndarray) -> list[FaceBox]:
        """Ejecuta la detección de caras con YuNet.

        Args:
            image: Imagen original BGR (YuNet requiere 3 canales).

        Returns:
            Lista de FaceBox con las coordenadas de cada cara detectada.
        """
        height, width = image.shape[:2]
        self._detector.setInputSize((width, height))
        faces = self._detector.detect(image)

        if faces[1] is None or len(faces[1]) == 0:
            return []

        return [
            FaceBox(x=int(face[0]), y=int(face[1]), width=int(face[2]), height=int(face[3]))
            for face in faces[1]
        ]


class DlibDetector(FaceDetector):
    """Detector basado en dlib (HOG o CNN).

    Args:
        use_cnn: Si True usa el modelo CNN (más preciso, más lento). Default False (HOG).
    """

    def __init__(self, use_cnn: bool = False) -> None:
        try:
            import dlib  # pyright: ignore[reportMissingImports]
        except ImportError as exc:
            raise ImportError("dlib no está instalado. Ejecuta: pip install dlib") from exc

        self._use_cnn = use_cnn

        if use_cnn:
            self._detector = dlib.cnn_face_detection_model_v1()  # pyright: ignore[reportAttributeAccessIssue]
        else:
            self._detector = dlib.get_frontal_face_detector()  # pyright: ignore[reportAttributeAccessIssue]

    @property
    def name(self) -> str:
        return DetectionStrategy.DLIB.value

    def detect(self, image: np.ndarray) -> list[FaceBox]:
        """Ejecuta la detección de caras con dlib.

        Args:
            image: Imagen original BGR o escala de grises.

        Returns:
            Lista de FaceBox con las coordenadas de cada cara detectada.
        """
        gray: np.ndarray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if self._use_cnn:
            dets = self._detector(gray, 1)
        else:
            dets = self._detector(gray)

        return [
            FaceBox(x=d.left(), y=d.top(), width=d.right() - d.left(), height=d.bottom() - d.top())
            for d in dets
        ]


# Registry de estrategias
_STRATEGY_REGISTRY: dict[str, type[FaceDetector]] = {
    DetectionStrategy.HAAR.value: HaarDetector,
    DetectionStrategy.YUNET.value: YuNetDetector,
    DetectionStrategy.DLIB.value: DlibDetector,
}


def get_detector(strategy: str, **kwargs: Any) -> FaceDetector:
    """Resuelve un detector de cara por nombre de estrategia.

    Args:
        strategy: Nombre de la estrategia (haar, yunet, dlib).
        **kwargs: Argumentos adicionales pasados al constructor del detector.
            Para 'yunet': model_path, input_size, conf_threshold, nms_threshold.
            Para 'dlib': use_cnn.

    Returns:
        Una instancia del detector solicitado.

    Raises:
        ValueError: Si la estrategia no está soportada.
    """
    strategy_lower = strategy.lower()
    detector_class = _STRATEGY_REGISTRY.get(strategy_lower)

    if detector_class is None:
        available = ", ".join(_STRATEGY_REGISTRY.keys())
        raise ValueError(f"Estrategia '{strategy}' no soportada. Disponibles: {available}")

    return detector_class(**kwargs)
