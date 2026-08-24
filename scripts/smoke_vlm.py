#!/usr/bin/env python3
"""Download/cache SmolVLM and verify one structured affect prediction."""

import json
from pathlib import Path

from PIL import Image

from affective_twins.io import write_json
from affective_twins.models import SmolVLMAdapter


PROJECT = Path(__file__).resolve().parents[1]
IMAGE = PROJECT.parent / "Emotional_colorTransfer" / "implementation" / "Emotion6" / "joy" / "14.jpg"
OUTPUT = PROJECT / "runs" / "vlm_smoke" / "smoke_prediction.json"


def main():
    adapter = SmolVLMAdapter("HuggingFaceTB/SmolVLM-500M-Instruct", local_files_only=True)
    with Image.open(IMAGE) as source:
        prediction = adapter.predict([source.convert("RGB")])[0]
    payload = {"image": str(IMAGE), "model": "HuggingFaceTB/SmolVLM-500M-Instruct", "prediction": prediction.to_dict()}
    write_json(OUTPUT, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
