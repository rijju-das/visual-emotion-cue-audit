"""Command-line interface for the complete framework."""

import argparse
import json
from pathlib import Path

from .config import PROJECT_ROOT, load_config
from .human import summarise_annotations
from .report import write_report
from .runner import doctor, evaluate, generate, train


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(prog="affective-twins")
    command.add_argument("command", choices=["doctor", "train", "generate", "evaluate", "report", "human-summary", "all"])
    command.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "emotion6_full.json")
    command.add_argument("--annotations", type=Path)
    return command


def main() -> None:
    args = parser().parse_args()
    config = load_config(args.config.resolve())
    if args.command == "doctor":
        result = doctor(config)
    elif args.command == "train":
        result = train(config)
    elif args.command == "generate":
        result = generate(config)
    elif args.command == "evaluate":
        result = evaluate(config)
    elif args.command == "report":
        summary_path = Path(config["run"]["output_dir"]) / "summary.json"
        result = {"report": str(write_report(summary_path.parent, json.loads(summary_path.read_text())))}
    elif args.command == "human-summary":
        annotation_path = args.annotations or Path(config["run"]["output_dir"]) / "human_validation_template.csv"
        result = summarise_annotations(annotation_path)
    else:
        if not Path(config["model"]["checkpoint"]).exists():
            train(config)
        generate(config)
        summary = evaluate(config)
        result = {**summary, "report": str(write_report(Path(config["run"]["output_dir"]), summary))}
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
