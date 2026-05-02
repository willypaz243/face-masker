"""Punto de entrada CLI — detección y enmascaramiento facial.

Este script es el entry point del paquete. Toda la lógica está en scripts/core/.
"""

from __future__ import annotations

import argparse
import sys

from scripts.core import (
    DetectionStrategy,
    generate_output_dir,
    get_detector,
    process_image,
)


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
    parser.add_argument(
        "--strategy",
        default=DetectionStrategy.HAAR.value,
        help="Estrategia de detección: haar, yunet, dlib (default: haar)",
    )
    parser.add_argument(
        "--yunet-model",
        default="",
        help="Ruta al archivo .onnx del modelo YuNet (solo para --strategy yunet)",
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

    try:
        detector_kwargs: dict = {}
        if args.strategy == DetectionStrategy.YUNET.value and args.yunet_model:
            detector_kwargs["model_path"] = args.yunet_model

        detector = get_detector(args.strategy, **detector_kwargs)
    except (ValueError, RuntimeError, ImportError) as exc:
        print(f"Error de configuración: {exc}", file=sys.stderr)
        return 1

    final_output_dir: str = generate_output_dir(args.output_dir)

    try:
        report = process_image(
            input_path=args.input,
            output_dir=final_output_dir,
            detector=detector,
        )
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
