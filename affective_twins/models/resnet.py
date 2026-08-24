"""Independent frozen-backbone models for emotion and valence-arousal."""

import math
import random
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18

from ..schema import AffectPrediction, AffectSample, EMOTIONS


class MultiTaskHead(nn.Module):
    def __init__(self, feature_dim: int = 512):
        super().__init__()
        self.emotion = nn.Linear(feature_dim, len(EMOTIONS))
        self.va = nn.Sequential(nn.Linear(feature_dim, 128), nn.ReLU(), nn.Dropout(0.15), nn.Linear(128, 2), nn.Tanh())

    def forward(self, features):
        return self.emotion(features), self.va(features)


class SampleDataset(Dataset):
    def __init__(self, samples: Sequence[AffectSample], transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        with Image.open(sample.image_path) as source:
            image = self.transform(source.convert("RGB"))
        va = [sample.valence if sample.valence is not None else 0.0, sample.arousal if sample.arousal is not None else 0.0]
        valid = float(sample.valence is not None and sample.arousal is not None)
        emotion_target = torch.tensor(
            [float(sample.emotion_distribution.get(label, 0.0)) for label in EMOTIONS],
            dtype=torch.float32,
        )
        if float(emotion_target.sum()) <= 0:
            emotion_target = torch.tensor(
                [float(label == sample.emotion) for label in EMOTIONS],
                dtype=torch.float32,
            )
        else:
            emotion_target = emotion_target / emotion_target.sum()
        return image, emotion_target, torch.tensor(va), valid


def _device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def _backbone(device):
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Identity()
    model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, ResNet18_Weights.DEFAULT.transforms()


def _extract(backbone, samples, transform, batch_size, device):
    loader = DataLoader(SampleDataset(samples, transform), batch_size=batch_size, shuffle=False, num_workers=0)
    features, labels, va_targets, va_valid = [], [], [], []
    with torch.inference_mode():
        for images, batch_labels, batch_va, batch_valid in loader:
            features.append(backbone(images.to(device)).cpu())
            labels.append(batch_labels)
            va_targets.append(batch_va.float())
            va_valid.append(batch_valid.bool())
    return torch.cat(features), torch.cat(labels), torch.cat(va_targets), torch.cat(va_valid)


def _macro_f1(logits, labels):
    predictions = logits.argmax(1)
    labels = labels.argmax(1) if labels.ndim == 2 else labels
    scores = []
    for index in range(len(EMOTIONS)):
        pred, truth = predictions == index, labels == index
        tp = (pred & truth).sum().float()
        precision = tp / (pred.sum().float().clamp_min(1.0))
        recall = tp / (truth.sum().float().clamp_min(1.0))
        scores.append(2 * precision * recall / (precision + recall).clamp_min(1e-8))
    return float(torch.stack(scores).mean())


def _train_head(train_x, train_y, train_va, train_valid, val_x, val_y, val_va, val_valid, seed, epochs=250):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    bootstrap = torch.randint(0, len(train_x), (len(train_x),), generator=generator)
    head = MultiTaskHead(train_x.shape[1])
    optimiser = torch.optim.AdamW(head.parameters(), lr=3e-3, weight_decay=1e-3)
    best_state, best_score, stale = None, -math.inf, 0
    for _ in range(epochs):
        head.train()
        optimiser.zero_grad()
        emotion_logits, va = head(train_x[bootstrap])
        valid = train_valid[bootstrap]
        emotion_loss = nn.functional.cross_entropy(emotion_logits, train_y[bootstrap])
        va_loss = nn.functional.smooth_l1_loss(va[valid], train_va[bootstrap][valid]) if valid.any() else torch.tensor(0.0)
        (emotion_loss + 0.65 * va_loss).backward()
        optimiser.step()
        head.eval()
        with torch.no_grad():
            val_logits, val_pred_va = head(val_x)
            f1 = _macro_f1(val_logits, val_y)
            va_mae = float((val_pred_va[val_valid] - val_va[val_valid]).abs().mean()) if val_valid.any() else 1.0
            score = f1 - 0.15 * va_mae
        if score > best_score + 1e-6:
            best_score, best_state, stale = score, deepcopy(head.state_dict()), 0
        else:
            stale += 1
        if stale >= 40:
            break
    head.load_state_dict(best_state)
    head.eval()
    return head, best_score


def train_independent_models(samples: List[AffectSample], checkpoint: Path, seed: int = 42, batch_size: int = 64) -> Dict[str, float]:
    device = _device()
    backbone, transform = _backbone(device)
    train = [sample for sample in samples if sample.split == "train"]
    validation = [sample for sample in samples if sample.split == "validation"]
    test = [sample for sample in samples if sample.split == "test"]
    train_data = _extract(backbone, train, transform, batch_size, device)
    val_data = _extract(backbone, validation, transform, batch_size, device)
    test_data = _extract(backbone, test, transform, batch_size, device)
    mean = train_data[0].mean(0, keepdim=True)
    std = train_data[0].std(0, keepdim=True).clamp_min(1e-6)
    train_x, val_x, test_x = (train_data[0] - mean) / std, (val_data[0] - mean) / std, (test_data[0] - mean) / std
    locator, locator_score = _train_head(train_x, *train_data[1:], val_x, *val_data[1:], seed + 101)
    evaluator, evaluator_score = _train_head(train_x, *train_data[1:], val_x, *val_data[1:], seed + 202)
    with torch.no_grad():
        logits, va = evaluator(test_x)
        test_accuracy = float((logits.argmax(1) == test_data[1].argmax(1)).float().mean())
        valid = test_data[3]
        test_va_mae = float((va[valid] - test_data[2][valid]).abs().mean()) if valid.any() else float("nan")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format_version": 3,
        "label_source": "human_distribution",
        "locator_head": locator.state_dict(),
        "evaluator_head": evaluator.state_dict(),
        "feature_mean": mean,
        "feature_std": std,
        "class_names": EMOTIONS,
        "seed": seed,
        "metrics": {"locator_validation_score": locator_score, "evaluator_validation_score": evaluator_score, "evaluator_test_accuracy": test_accuracy, "evaluator_test_va_mae": test_va_mae},
    }, checkpoint)
    return {"locator_validation_score": locator_score, "evaluator_validation_score": evaluator_score, "evaluator_test_accuracy": test_accuracy, "evaluator_test_va_mae": test_va_mae}


class ResNetAffectModel:
    def __init__(self, checkpoint: Path, role: str = "evaluator"):
        self.device = _device()
        self.backbone, self.transform = _backbone(self.device)
        payload = torch.load(checkpoint, map_location="cpu")
        if payload.get("format_version") != 3 or payload.get("label_source") != "human_distribution":
            raise ValueError(
                "Checkpoint was not trained with corrected human-distribution labels; "
                "run `affective-twins train` again"
            )
        self.head = MultiTaskHead()
        self.head.load_state_dict(payload["{}_head".format(role)])
        self.head.eval()
        self.mean = payload["feature_mean"]
        self.std = payload["feature_std"]

    def predict(self, images: Sequence[Image.Image]) -> Tuple[List[AffectPrediction], torch.Tensor]:
        if not images:
            return [], torch.empty((0, 512))
        tensors = torch.stack([self.transform(image.convert("RGB")) for image in images]).to(self.device)
        with torch.inference_mode():
            raw_features = self.backbone(tensors).cpu()
            logits, va = self.head((raw_features - self.mean) / self.std)
            probabilities = logits.softmax(1)
        results = []
        for index in range(len(images)):
            probs = {label: float(probabilities[index, class_index]) for class_index, label in enumerate(EMOTIONS)}
            predicted = max(probs, key=probs.get)
            results.append(AffectPrediction(probs, float(va[index, 0]), float(va[index, 1]), probs[predicted], predicted))
        return results, raw_features
