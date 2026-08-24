"""Configuration loading with paths anchored to the project directory."""

import json
from pathlib import Path
from typing import Any, Dict


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve(value: str) -> str:
    path = Path(value)
    return str(path if path.is_absolute() else (PROJECT_ROOT / path).resolve())


def load_config(path: Path) -> Dict[str, Any]:
    with path.open() as stream:
        config = json.load(stream)
    for section, keys in {
        "dataset": ["image_root", "ground_truth_csv", "auxiliary_manifest", "audit_manifest"],
        "run": ["output_dir"],
        "model": ["checkpoint", "vlm_cache_dir"],
        "assets": ["face_landmarker", "face_detector_prototxt", "face_detector_model", "tesseract"],
    }.items():
        for key in keys:
            if key in config.get(section, {}):
                config[section][key] = _resolve(config[section][key])
    return config
