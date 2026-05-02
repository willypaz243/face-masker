# face-masker

Herramienta CLI local de detección y enmascaramiento facial basada en Haar Cascade de OpenCV.

Recibe una imagen como entrada, detecta rostros frontales, genera una máscara con las zonas faciales y produce un reporte JSON con métricas de la detección.

## Instalación

### Con uv (recomendado)

[Instalar uv](https://docs.astral.sh/uv/getting-started/installation/) → [Guía de entornos](https://docs.astral.sh/uv/guides/environments/)

```bash
uv venv && uv sync
```

### Con pip

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Uso

```bash
# uv
uv run python scripts/detect_faces_and_mask.py --input path/to/photo.jpg

# pip (venv activo)
source .venv/bin/activate
python scripts/detect_faces_and_mask.py --input path/to/photo.jpg

# directorio de salida personalizado
python scripts/detect_faces_and_mask.py --input path/to/photo.jpg --output-dir outputs/my_folder
```

### Argumentos

| Argumento      | Requerido | Por defecto | Descripción                          |
| -------------- | --------- | ----------- | ------------------------------------ |
| `--input`      | Sí        | —           | Ruta a la imagen de entrada          |
| `--output-dir` | No        | `outputs`   | Directorio de salida (debe comenzar con `outputs/`) |

### Reglas de `--output-dir`

| Entrada | Resultado |
|---|---|
| `outputs` | Crea `outputs/YYYYMMDD_HHMMSS/` con timestamp |
| `outputs/mi_carpeta` | Usa `outputs/mi_carpeta` directamente |
| `carpeta/otra` | **Error**: debe comenzar con `outputs/` |

El directorio `outputs/` se crea automáticamente si no existe.

## Extra

### Estrategias de detección

El script soporta tres estrategias. Se seleccionan con `--strategy`:

| Estrategia        | Descripción                       | Requiere instalación extra                      |
| ----------------- | --------------------------------- | ----------------------------------------------- |
| `haar`            | Haar Cascade, embebido en OpenCV  | No                                              |
| `yunet` (default) | YuNet (OpenCV DNN), más preciso   | No                                              |
| `dlib`            | HOG + CNN de dlib, el más preciso | Sí (`pip install dlib` o `uv pip install dlib`) |

```bash
# Haar (por defecto)
uv run python scripts/detect_faces_and_mask.py --input path/to/photo.jpg

# YuNet
uv run python scripts/detect_faces_and_mask.py --input path/to/photo.jpg --strategy yunet

# Dlib (requiere: pip install dlib o uv pip install dlib)
uv run python scripts/detect_faces_and_mask.py --input path/to/photo.jpg --strategy dlib
```

## Outputs

Se generan tres archivos en el directorio de salida:

| Archivo              | Formato | Descripción                                                 |
| -------------------- | ------- | ----------------------------------------------------------- |
| `detected_faces.jpg` | JPG     | Imagen original con cajas verdes sobre las caras detectadas |
| `mask.png`           | PNG     | Máscara negra con zonas faciales en blanco                  |
| `report.json`        | JSON    | Reporte con métricas de la detección                        |

### Reporte JSON

```json
{
  "face_detected": true,
  "num_faces": 2,
  "bounding_boxes": [
    {"x": 120, "y": 80, "w": 150, "h": 150},
    {"x": 300, "y": 90, "w": 140, "h": 140}
  ],
  "image_size": [800, 600],
  "mask_coverage_pct": 3.56,
  "warnings": []
}
```

## Tests

### Ejecutar

```bash
uv run pytest tests/ -v
```

### Qué se prueba

| Categoría           | Qué cubre                                                                                          |
| ------------------- | -------------------------------------------------------------------------------------------------- |
| **Funciones puras** | `draw_result_image`, `create_mask_image`, `calculate_coverage` — formas, píxeles, cobertura        |
| **Reporte**         | `build_report` con y sin caras, estructura de `to_dict()`                                          |
| **Estrategias**     | Cada estrategia (`haar`, `yunet`, `dlib`) ejecuta detección + pipeline completo                    |
| **Errores**         | Archivo inexistente (`FileNotFoundError`), formato inválido (`ValueError`), estrategia desconocida |

### Ejecutar solo una estrategia

```bash
uv run pytest tests/ -v -k "haar"
uv run pytest tests/ -v -k "yunet"
pip install dlib # o uv pip install dlib
uv run pytest tests/ -v -k "dlib"
```

