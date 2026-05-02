# AGENTS.md — face-masker

## Quick facts
- **Package manager**: `uv` (primary). Venv at `.venv`. Run everything with `uv run <cmd>`.
- **Entry point**: `detect-faces` CLI (`scripts/detect_faces_and_mask.py`)
- **Test image**: `uv run python scripts/generate_test_image.py` → `test_synthetic.jpg`

## Tool config (verified from pyproject.toml)
- **Ruff**: double quotes, indent spaces, ignore E501, rules: E/F/W/I/N/B/UP/C9
- **basedpyright**: `typeCheckingMode = "basic"`, scans `scripts/` + `tests/` only, venv at `.venv`
- **Python**: >=3.10

## Quality gate order
```bash
uv run ruff format .        # format
uv run ruff check .          # lint
uv run basedpyright          # type check (must be 0 errors)
uv run pytest tests/ -v     # test
```

## Architecture
```
scripts/
├── detect_faces_and_mask.py   # CLI entry point (thin wrapper, ~60 lines)
├── generate_test_image.py     # Test image generator
├── __init__.py                # Backward compat re-exports for tests
└── core/
    ├── __init__.py            # Core re-exports
    ├── types.py               # FaceBox, DetectionReport, DetectionStrategy enum
    ├── detector.py            # FaceDetector base + Haar/YuNet/Dlib strategies + registry
    ├── image_ops.py           # draw_result_image(), create_mask_image()
    ├── metrics.py             # calculate_coverage(), build_report()
    ├── io_utils.py            # generate_output_dir(), write_outputs()
    └── pipeline.py            # process_image() orchestration
```

## CLI usage
```bash
# Haar (default)
uv run detect-faces --input test.jpg

# YuNet with custom model
uv run detect-faces --input test.jpg --strategy yunet --yunet-model /path/to/model.onnx

# Dlib (requires: pip install dlib)
uv run detect-faces --input test.jpg --strategy dlib
```

## Project-specific rules
- **Follow SOLID, KISS, and minimal design patterns** — prefer pure functions over classes, single responsibility per function, no over-engineering
- **Max 2 nesting levels** in any function — use early returns
- **`@dataclass(frozen=True)`** for immutable objects; `@dataclass` for mutable
- **OpenCV only** for image manipulation — never Pillow
- **Strict typing**: all variables, params, returns annotated; use `T | None` for optional
- **All public functions** need docstring with Args/Returns
- **No new dependencies** without justification

## Allowed runtime deps
`opencv-python>=4.10` only. `numpy` is a transitive dep (not listed in pyproject.toml).

## New strategy dependencies
| Strategy | Dep | Install |
|---|---|---|
| yunet | None (OpenCV DNN included) | — |
| dlib | `dlib>=20.0` | `pip install dlib` |
