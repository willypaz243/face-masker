# Face Masker

Herramienta local de detección y enmascaramiento facial basada en Haar Cascade de OpenCV.

## Descripción

Esta herramienta recibe una imagen, detecta una o varias caras, genera una máscara facial básica y produce un reporte JSON con métricas de la detección.

- Detección de rostros frontal con **Haar Cascade** (embebido en OpenCV)
- Generación de **bounding boxes** sobre las caras detectadas
- Creación de **máscara** con zonas faciales marcadas
- Reporte JSON con métricas completas
- Cero dependencias externas, cero descargas de modelos

## Instalación

### Opción 1: Con uv (recomendado)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
cd face-masker
uv venv
uv sync
```

### Opción 2: Con pip tradicional

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Uso

### Generar imagen de prueba

```bash
# Con uv
uv run python scripts/generate_test_image.py

# Con pip (venv activo)
python scripts/generate_test_image.py
```

Esto genera `test_synthetic.jpg` (500x500px con círculos decorativos).

### Ejecutar la herramienta de detección

```bash
# Con uv
uv run python scripts/detect_faces_and_mask.py --input test_synthetic.jpg --output-dir outputs/test1

# Con pip (venv activo)
python scripts/detect_faces_and_mask.py --input test_synthetic.jpg --output-dir outputs/test1

# Con imagen real
uv run python scripts/detect_faces_and_mask.py --input foto.jpg --output-dir outputs/real_test
```

### Argumentos

| Argumento | Requerido | Por defecto | Descripción |
|---|---|---|---|
| `--input` | Sí | — | Ruta a la imagen de entrada |
| `--output-dir` | No | `outputs/face_detection` | Directorio donde guardar los outputs |

## Outputs

La herramienta genera en el directorio de salida:

| Archivo | Formato | Descripción |
|---|---|---|
| `detected_faces.jpg` | JPG | Imagen original con cajas verdes sobre las caras |
| `mask.png` | PNG | Máscara negra con zonas faciales en blanco |
| `report.json` | JSON | Reporte con métricas de la detección |

### Estructura del reporte JSON

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

| Campo | Tipo | Descripción |
|---|---|---|
| `face_detected` | boolean | Si al menos una cara fue detectada |
| `num_faces` | int | Cantidad de caras detectadas |
| `bounding_boxes` | array | Lista de cajas `[x, y, w, h]` por cara |
| `image_size` | array | `[ancho, alto]` de la imagen original |
| `mask_coverage_pct` | float | Porcentaje de área cubierta por la máscara |
| `warnings` | array | Lista de advertencias |

## Ejecutar tests

```bash
# Con uv
uv run pytest tests/ -v

# Con pip (venv activo)
pytest tests/ -v
```

## Calidad de código

```bash
# Formateo con Ruff
ruff format .

# Linting con Ruff
ruff check .

# Type checking con basedpyright
basedpyright
```

## Principios aplicados

- **SOLID**: Responsabilidades separadas, funciones puras, datos inmutables con `frozen=True`
- **KISS**: Cero clases innecesarias, flujo lineal, configuración simple
- **Máximo 2 niveles de anidación**: Early returns y extracción de funciones
- **Tipado estricto**: Todas las variables, parámetros y propiedades con tipo explícito

## Limitaciones

- Haar Cascade detecta **rostros frontales** con buena iluminación
- No funciona bien con perfiles laterales, objetos o mascotas
- La precisión es menor que modelos basados en deep learning (MTCNN, YOLO)
- El clasificador XML viene embebido en `opencv-python` (cero descargas externas)

## Estructura del proyecto

```
face-masker/
├── pyproject.toml          # Dependencias y configuración de herramientas
├── requirements.txt        # Compatibilidad con pip
├── scripts/
│   ├── detect_faces_and_mask.py   # CLI principal
│   └── generate_test_image.py     # Generador de imagen sintética
├── tests/
│   └── test_detection.py          # Tests automatizados
├── outputs/                    # Carpeta generada al ejecutar
└── README.md
```
