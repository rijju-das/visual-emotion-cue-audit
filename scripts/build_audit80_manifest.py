#!/usr/bin/env python3
"""Copy the exact 80 audit originals into a portable project-local manifest."""

import json
import shutil
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
SOURCE_MANIFEST = PROJECT / "runs" / "emotion6_abaw80_vlm" / "samples.jsonl"
OUTPUT = PROJECT / "data" / "audit80"


def main():
    rows = [json.loads(line) for line in SOURCE_MANIFEST.read_text().splitlines() if line.strip()]
    if len(rows) != 80:
        raise RuntimeError("Expected 80 source samples, found {}".format(len(rows)))
    image_dir = OUTPUT / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    portable = []
    for row in rows:
        source = Path(row["image_path"])
        suffix = source.suffix.lower() or ".jpg"
        destination = image_dir / "{}{}".format(row["sample_id"], suffix)
        shutil.copy2(str(source), str(destination))
        row["image_path"] = str(Path("images") / destination.name)
        portable.append(row)
    manifest = OUTPUT / "manifest.jsonl"
    with manifest.open("w") as stream:
        for row in portable:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    summary = {
        "n": len(portable),
        "emotion6": sum(row.get("metadata", {}).get("source_dataset") != "Aff-Wild2/ABAW" for row in portable),
        "abaw_au": sum(row.get("metadata", {}).get("source_dataset") == "Aff-Wild2/ABAW" for row in portable),
        "manifest": str(manifest),
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
