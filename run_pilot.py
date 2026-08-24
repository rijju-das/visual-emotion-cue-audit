#!/usr/bin/env python3
"""Run a compact counterfactual-colour pilot for the WiML 2026 abstract.

The experiment is deliberately self-contained and uses only dependencies that
already exist in the workspace: PyTorch, torchvision, Pillow, and NumPy.
"""

import argparse
import csv
import json
import math
import random
from copy import deepcopy
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18


CLASS_NAMES = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cross-model audit of region-specific chroma interventions."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("../Emotional_colorTransfer/implementation/Emotion6"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument("--pilot-per-class", type=int, default=10)
    parser.add_argument("--grid-size", type=int, default=4)
    parser.add_argument("--random-controls", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def discover_images(root):
    records = []
    missing = []
    for label, class_name in enumerate(CLASS_NAMES):
        class_dir = root / class_name
        if not class_dir.is_dir():
            missing.append(str(class_dir))
            continue
        paths = sorted(
            path
            for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        records.extend((path, label) for path in paths)
    if missing:
        raise FileNotFoundError("Missing class directories: " + ", ".join(missing))
    if not records:
        raise RuntimeError("No images were found under {}".format(root))
    return records


def stratified_split(records, train_fraction, val_fraction, seed):
    by_class = {idx: [] for idx in range(len(CLASS_NAMES))}
    for path, label in records:
        by_class[label].append((path, label))

    rng = random.Random(seed)
    train, val, test = [], [], []
    for label in range(len(CLASS_NAMES)):
        items = list(by_class[label])
        rng.shuffle(items)
        n_train = int(len(items) * train_fraction)
        n_val = int(len(items) * val_fraction)
        train.extend(items[:n_train])
        val.extend(items[n_train : n_train + n_val])
        test.extend(items[n_train + n_val :])
    return train, val, test


class ImageRecords(Dataset):
    def __init__(self, records, transform):
        self.records = records
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        path, label = self.records[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
            tensor = self.transform(image)
        return tensor, label, str(path)


def build_backbone(device):
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.fc = nn.Identity()
    model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, weights.transforms()


def extract_features(backbone, records, transform, batch_size, device):
    dataset = ImageRecords(records, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    feature_batches, label_batches, paths = [], [], []
    with torch.inference_mode():
        for images, labels, batch_paths in loader:
            feature_batches.append(backbone(images.to(device)).cpu())
            label_batches.append(labels.cpu())
            paths.extend(batch_paths)
    return torch.cat(feature_batches), torch.cat(label_batches), paths


def standardise_features(train_x, *others):
    mean = train_x.mean(dim=0, keepdim=True)
    std = train_x.std(dim=0, keepdim=True).clamp_min(1e-6)
    outputs = [(train_x - mean) / std]
    outputs.extend((item - mean) / std for item in others)
    return outputs, mean, std


class LinearEmotionHead(nn.Module):
    def __init__(self, feature_dim, num_classes):
        super().__init__()
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, features):
        return self.classifier(features)


def accuracy(logits, labels):
    return float((logits.argmax(dim=1) == labels).float().mean().item())


def macro_f1(logits, labels, num_classes):
    predictions = logits.argmax(dim=1)
    scores = []
    for class_idx in range(num_classes):
        pred_pos = predictions == class_idx
        true_pos = labels == class_idx
        tp = (pred_pos & true_pos).sum().float()
        fp = (pred_pos & ~true_pos).sum().float()
        fn = (~pred_pos & true_pos).sum().float()
        precision = tp / (tp + fp).clamp_min(1.0)
        recall = tp / (tp + fn).clamp_min(1.0)
        score = 2.0 * precision * recall / (precision + recall).clamp_min(1e-8)
        scores.append(score)
    return float(torch.stack(scores).mean().item())


def train_head(train_x, train_y, val_x, val_y, seed, epochs, lr, weight_decay):
    seed_everything(seed)
    generator = torch.Generator().manual_seed(seed)
    bootstrap = torch.randint(
        0, train_x.shape[0], (train_x.shape[0],), generator=generator
    )
    boot_x = train_x[bootstrap]
    boot_y = train_y[bootstrap]

    head = LinearEmotionHead(train_x.shape[1], len(CLASS_NAMES))
    optimiser = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    best_state = None
    best_f1 = -math.inf
    patience = 35
    stale_epochs = 0

    for _ in range(epochs):
        head.train()
        optimiser.zero_grad()
        loss = nn.functional.cross_entropy(head(boot_x), boot_y)
        loss.backward()
        optimiser.step()

        head.eval()
        with torch.no_grad():
            val_logits = head(val_x)
            val_f1 = macro_f1(val_logits, val_y, len(CLASS_NAMES))
        if val_f1 > best_f1 + 1e-6:
            best_f1 = val_f1
            best_state = deepcopy(head.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= patience:
            break

    head.load_state_dict(best_state)
    head.eval()
    return head, best_f1


def choose_pilot_records(test_records, per_class, seed):
    by_class = {idx: [] for idx in range(len(CLASS_NAMES))}
    for record in test_records:
        by_class[record[1]].append(record)
    rng = random.Random(seed)
    selected = []
    for label in range(len(CLASS_NAMES)):
        items = list(by_class[label])
        rng.shuffle(items)
        selected.extend(items[: min(per_class, len(items))])
    rng.shuffle(selected)
    return selected


def chroma_remove_cell(image, cell_index, grid_size):
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    height, width = rgb.shape[:2]
    row, col = divmod(cell_index, grid_size)
    y0 = int(round(row * height / grid_size))
    y1 = int(round((row + 1) * height / grid_size))
    x0 = int(round(col * width / grid_size))
    x1 = int(round((col + 1) * width / grid_size))
    out = rgb.copy()
    region = out[y0:y1, x0:x1]
    luminance = (
        0.2126 * region[..., 0]
        + 0.7152 * region[..., 1]
        + 0.0722 * region[..., 2]
    )
    region[:] = luminance[..., None]
    return Image.fromarray(np.clip(out * 255.0, 0, 255).astype(np.uint8))


def image_batch_features(backbone, images, transform, device):
    tensors = torch.stack([transform(image) for image in images]).to(device)
    with torch.inference_mode():
        return backbone(tensors).cpu()


def normalise(features, mean, std):
    return (features - mean) / std


def cosine_similarity(a, b):
    a = a / a.norm(dim=1, keepdim=True).clamp_min(1e-8)
    b = b / b.norm(dim=1, keepdim=True).clamp_min(1e-8)
    return (a * b).sum(dim=1)


def entropy_from_logits(logits):
    probabilities = logits.softmax(dim=1)
    return -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=1)


def paired_bootstrap(values, replicates, seed):
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        sample = rng.integers(0, len(values), size=len(values))
        means[index] = values[sample].mean()
    return {
        "mean": float(values.mean()),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
    }


def run_counterfactual_audit(
    backbone,
    locator_head,
    evaluator_head,
    records,
    transform,
    feature_mean,
    feature_std,
    grid_size,
    random_controls,
    seed,
    device,
):
    rng = random.Random(seed)
    rows = []
    examples = []
    num_cells = grid_size * grid_size

    for path, label in records:
        with Image.open(path) as source:
            original = source.convert("RGB")
        variants = [chroma_remove_cell(original, idx, grid_size) for idx in range(num_cells)]
        images = [original] + variants
        raw_features = image_batch_features(backbone, images, transform, device)
        features = normalise(raw_features, feature_mean, feature_std)

        with torch.no_grad():
            locator_logits = locator_head(features)
            evaluator_logits = evaluator_head(features)

        locator_probs = locator_logits.softmax(dim=1)[:, label]
        evaluator_probs = evaluator_logits.softmax(dim=1)[:, label]
        locator_drops = locator_probs[0] - locator_probs[1:]
        causal_cell = int(locator_drops.argmax().item())

        control_candidates = [idx for idx in range(num_cells) if idx != causal_cell]
        control_cells = rng.sample(
            control_candidates, min(random_controls, len(control_candidates))
        )
        causal_index = causal_cell + 1
        control_indices = [cell + 1 for cell in control_cells]

        base_prob = float(evaluator_probs[0].item())
        causal_prob = float(evaluator_probs[causal_index].item())
        control_probs = [float(evaluator_probs[idx].item()) for idx in control_indices]
        causal_drop = base_prob - causal_prob
        control_drop = base_prob - float(np.mean(control_probs))

        base_prediction = int(evaluator_logits[0].argmax().item())
        causal_prediction = int(evaluator_logits[causal_index].argmax().item())
        control_predictions = [
            int(evaluator_logits[idx].argmax().item()) for idx in control_indices
        ]
        base_entropy = float(entropy_from_logits(evaluator_logits[0:1])[0].item())
        causal_entropy = float(
            entropy_from_logits(evaluator_logits[causal_index : causal_index + 1])[0].item()
        )
        control_entropy = float(
            entropy_from_logits(evaluator_logits[control_indices]).mean().item()
        )
        base_raw = raw_features[0:1]
        causal_cosine = float(
            cosine_similarity(base_raw, raw_features[causal_index : causal_index + 1])[0].item()
        )
        control_cosine = float(
            cosine_similarity(
                base_raw.repeat(len(control_indices), 1), raw_features[control_indices]
            ).mean().item()
        )

        row = {
            "image": str(path),
            "label": CLASS_NAMES[label],
            "base_prediction": CLASS_NAMES[base_prediction],
            "causal_prediction": CLASS_NAMES[causal_prediction],
            "control_predictions": ";".join(CLASS_NAMES[idx] for idx in control_predictions),
            "causal_cell": causal_cell,
            "control_cells": ";".join(str(idx) for idx in control_cells),
            "base_true_probability": base_prob,
            "causal_true_probability": causal_prob,
            "control_true_probability_mean": float(np.mean(control_probs)),
            "display_control_true_probability": control_probs[0],
            "causal_probability_drop": causal_drop,
            "control_probability_drop": control_drop,
            "display_control_probability_drop": base_prob - control_probs[0],
            "paired_drop_advantage": causal_drop - control_drop,
            "causal_entropy_change": causal_entropy - base_entropy,
            "control_entropy_change": control_entropy - base_entropy,
            "causal_feature_cosine": causal_cosine,
            "control_feature_cosine": control_cosine,
            "base_correct": int(base_prediction == label),
            "causal_flipped_from_base": int(causal_prediction != base_prediction),
            "control_flip_rate": float(
                np.mean([prediction != base_prediction for prediction in control_predictions])
            ),
        }
        rows.append(row)
        examples.append(
            {
                "row": row,
                "original": original,
                "causal": variants[causal_cell],
                "control": variants[control_cells[0]],
            }
        )
    return rows, examples


def aggregate_results(rows, replicates, seed):
    advantages = [row["paired_drop_advantage"] for row in rows]
    correct_rows = [row for row in rows if row["base_correct"] == 1]
    correct_advantages = [row["paired_drop_advantage"] for row in correct_rows]
    per_class = {}
    for class_name in CLASS_NAMES:
        class_rows = [row for row in rows if row["label"] == class_name]
        class_advantages = [row["paired_drop_advantage"] for row in class_rows]
        per_class[class_name] = {
            "n": len(class_rows),
            "base_accuracy": float(np.mean([row["base_correct"] for row in class_rows])),
            "causal_probability_drop_mean": float(
                np.mean([row["causal_probability_drop"] for row in class_rows])
            ),
            "control_probability_drop_mean": float(
                np.mean([row["control_probability_drop"] for row in class_rows])
            ),
            "paired_drop_advantage_mean": float(np.mean(class_advantages)),
            "causal_greater_than_control_rate": float(
                np.mean([value > 0 for value in class_advantages])
            ),
        }
    return {
        "n_images": len(rows),
        "n_base_correct": len(correct_rows),
        "base_accuracy_on_pilot": float(np.mean([row["base_correct"] for row in rows])),
        "causal_probability_drop_mean": float(
            np.mean([row["causal_probability_drop"] for row in rows])
        ),
        "control_probability_drop_mean": float(
            np.mean([row["control_probability_drop"] for row in rows])
        ),
        "causal_greater_than_control_rate": float(
            np.mean([value > 0 for value in advantages])
        ),
        "paired_drop_advantage": paired_bootstrap(advantages, replicates, seed),
        "paired_drop_advantage_base_correct": (
            paired_bootstrap(correct_advantages, replicates, seed + 1)
            if correct_advantages
            else None
        ),
        "causal_flip_rate": float(
            np.mean([row["causal_flipped_from_base"] for row in rows])
        ),
        "control_flip_rate": float(np.mean([row["control_flip_rate"] for row in rows])),
        "causal_entropy_change_mean": float(
            np.mean([row["causal_entropy_change"] for row in rows])
        ),
        "control_entropy_change_mean": float(
            np.mean([row["control_entropy_change"] for row in rows])
        ),
        "causal_feature_cosine_mean": float(
            np.mean([row["causal_feature_cosine"] for row in rows])
        ),
        "control_feature_cosine_mean": float(
            np.mean([row["control_feature_cosine"] for row in rows])
        ),
        "per_class": per_class,
    }


def font(size):
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def draw_grid_outline(image, cell_index, grid_size, colour=(255, 255, 255)):
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    width, height = canvas.size
    row, col = divmod(cell_index, grid_size)
    x0 = int(round(col * width / grid_size))
    x1 = int(round((col + 1) * width / grid_size))
    y0 = int(round(row * height / grid_size))
    y1 = int(round((row + 1) * height / grid_size))
    draw.rectangle((x0, y0, x1 - 1, y1 - 1), outline=colour, width=4)
    return canvas


def make_figure(examples, output_path, grid_size):
    correct_examples = [item for item in examples if item["row"]["base_correct"] == 1]
    candidate_examples = correct_examples if len(correct_examples) >= 3 else examples
    ranked = sorted(
        candidate_examples,
        key=lambda item: item["row"]["paired_drop_advantage"],
        reverse=True,
    )[:1]
    panel_width, image_height, caption_height = 320, 260, 78
    header_height = 58
    canvas = Image.new(
        "RGB",
        (panel_width * 3, header_height + len(ranked) * (image_height + caption_height)),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    title_font = font(19)
    body_font = font(18)
    small_font = font(15)
    headers = ["Original", "Selected intervention", "Other equal-area region"]
    for col, header in enumerate(headers):
        box = draw.textbbox((0, 0), header, font=title_font)
        text_width = box[2] - box[0]
        draw.text(
            (col * panel_width + (panel_width - text_width) // 2, 18),
            header,
            fill="black",
            font=title_font,
        )

    for row_idx, item in enumerate(ranked):
        row = item["row"]
        y = header_height + row_idx * (image_height + caption_height)
        control_cell = int(row["control_cells"].split(";")[0])
        panels = [
            item["original"],
            draw_grid_outline(item["causal"], int(row["causal_cell"]), grid_size),
            draw_grid_outline(item["control"], control_cell, grid_size),
        ]
        for col, panel in enumerate(panels):
            fitted = panel.copy()
            fitted.thumbnail((panel_width, image_height), Image.Resampling.LANCZOS)
            x = col * panel_width + (panel_width - fitted.width) // 2
            canvas.paste(fitted, (x, y + (image_height - fitted.height) // 2))

        captions = [
            "label: {}\np={:.3f}, pred={}".format(
                row["label"], row["base_true_probability"], row["base_prediction"]
            ),
            "p={:.3f}, drop={:+.3f}\npred={}".format(
                row["causal_true_probability"],
                row["causal_probability_drop"],
                row["causal_prediction"],
            ),
            "p={:.3f}, drop={:+.3f}\npred={}".format(
                row["display_control_true_probability"],
                row["display_control_probability_drop"],
                row["control_predictions"].split(";")[0],
            ),
        ]
        for col, caption in enumerate(captions):
            text_y = y + image_height + 6
            for line_index, line in enumerate(caption.splitlines()):
                draw.text(
                    (col * panel_width + 12, text_y + line_index * 23),
                    line,
                    fill="black",
                    font=body_font if line_index == 0 else small_font,
                )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not 0 < args.train_fraction < 1:
        raise ValueError("--train-fraction must be between 0 and 1")
    if not 0 <= args.val_fraction < 1:
        raise ValueError("--val-fraction must be between 0 and 1")
    if args.train_fraction + args.val_fraction >= 1:
        raise ValueError("train and validation fractions must sum to less than 1")

    records = discover_images(args.dataset_root)
    train_records, val_records, test_records = stratified_split(
        records, args.train_fraction, args.val_fraction, args.seed
    )
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    backbone, transform = build_backbone(device)

    print("Extracting frozen ResNet-18 features...")
    train_x, train_y, _ = extract_features(
        backbone, train_records, transform, args.batch_size, device
    )
    val_x, val_y, _ = extract_features(
        backbone, val_records, transform, args.batch_size, device
    )
    test_x, test_y, _ = extract_features(
        backbone, test_records, transform, args.batch_size, device
    )
    (train_x, val_x, test_x), feature_mean, feature_std = standardise_features(
        train_x, val_x, test_x
    )

    print("Training independent locator and evaluator heads...")
    locator_head, locator_val_f1 = train_head(
        train_x,
        train_y,
        val_x,
        val_y,
        args.seed + 101,
        args.epochs,
        args.learning_rate,
        args.weight_decay,
    )
    evaluator_head, evaluator_val_f1 = train_head(
        train_x,
        train_y,
        val_x,
        val_y,
        args.seed + 202,
        args.epochs,
        args.learning_rate,
        args.weight_decay,
    )
    with torch.no_grad():
        locator_test_logits = locator_head(test_x)
        evaluator_test_logits = evaluator_head(test_x)

    pilot_records = choose_pilot_records(
        test_records, args.pilot_per_class, args.seed + 303
    )
    print("Running {} counterfactual audits...".format(len(pilot_records)))
    rows, examples = run_counterfactual_audit(
        backbone,
        locator_head,
        evaluator_head,
        pilot_records,
        transform,
        feature_mean,
        feature_std,
        args.grid_size,
        args.random_controls,
        args.seed + 404,
        device,
    )

    summary = {
        "experiment": "cross-model region-specific chroma removal",
        "dataset": str(args.dataset_root),
        "class_names": CLASS_NAMES,
        "split_sizes": {
            "train": len(train_records),
            "validation": len(val_records),
            "test": len(test_records),
        },
        "backbone": "torchvision ResNet-18 ImageNet-1K V1, frozen",
        "locator": {
            "validation_macro_f1": locator_val_f1,
            "test_accuracy": accuracy(locator_test_logits, test_y),
            "test_macro_f1": macro_f1(locator_test_logits, test_y, len(CLASS_NAMES)),
        },
        "evaluator": {
            "validation_macro_f1": evaluator_val_f1,
            "test_accuracy": accuracy(evaluator_test_logits, test_y),
            "test_macro_f1": macro_f1(evaluator_test_logits, test_y, len(CLASS_NAMES)),
        },
        "intervention": {
            "grid_size": args.grid_size,
            "area_fraction": 1.0 / (args.grid_size * args.grid_size),
            "operation": "remove chroma while preserving luminance in one grid cell",
            "random_controls_per_image": args.random_controls,
            "selection_model": "locator",
            "evaluation_model": "evaluator",
        },
        "pilot": aggregate_results(rows, args.bootstrap_replicates, args.seed + 505),
        "limitations": [
            "The locator and evaluator share a frozen visual backbone and training dataset.",
            "Grid cells provide area matching but are not object-aligned regions.",
            "Emotion6 labels represent aggregate annotations rather than a single objective affective truth.",
        ],
    }

    with (args.output_dir / "results.json").open("w") as stream:
        json.dump(summary, stream, indent=2)
    write_csv(args.output_dir / "per_image.csv", rows)
    make_figure(examples, args.output_dir / "pilot_figure.png", args.grid_size)
    torch.save(
        {
            "locator_head": locator_head.state_dict(),
            "evaluator_head": evaluator_head.state_dict(),
            "feature_mean": feature_mean,
            "feature_std": feature_std,
            "class_names": CLASS_NAMES,
            "seed": args.seed,
            "split_records": {
                "train": [(str(path), label) for path, label in train_records],
                "validation": [(str(path), label) for path, label in val_records],
                "test": [(str(path), label) for path, label in test_records],
            },
        },
        args.output_dir / "models.pt",
    )

    print(json.dumps(summary, indent=2))
    print("Wrote outputs to {}".format(args.output_dir.resolve()))


if __name__ == "__main__":
    main()
