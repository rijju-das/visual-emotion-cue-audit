"""Constrained-choice adapter for the compact SmolVLM checkpoint."""

import json
import re
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
from PIL import Image

from ..schema import AffectPrediction, EMOTIONS


PROMPT = """Inspect the image as an affective-computing auditor. Return ONLY one JSON object with keys: emotion (one of anger, disgust, fear, joy, sadness, surprise), emotion_probabilities (object over those six labels summing to 1), valence (number -1 to 1), arousal (number -1 to 1), confidence (number 0 to 1), evidence_cue (one of color_lighting, facial_action_region, scene_context, embedded_text), evidence (short visible evidence phrase), caption (one literal sentence). Do not infer a person's private internal state; describe the apparent scene affect."""

EMOTION_PROMPT = "Classify the apparent emotion. Reply with exactly one word from: anger, disgust, fear, joy, sadness, surprise."
CUE_PROMPT = "Choose the strongest visible affect cue. Reply exactly one token from: color_lighting, facial_action_region, scene_context, embedded_text."
VALENCE_PROMPT = "Classify apparent valence. Reply exactly one word: negative, neutral, or positive."
AROUSAL_PROMPT = "Classify apparent arousal. Reply exactly one word: low, medium, or high."
CONFIDENCE_PROMPT = "How confident is the visible evidence for the apparent emotion? Reply exactly one word: low, medium, or high."
EVIDENCE_PROMPT = "Name the visible affect evidence in at most six words. Describe only what is visible; do not infer private internal state."


class SmolVLMAdapter:
    def __init__(self, model_name: str, local_files_only: bool = False, cache_dir: Optional[str] = None):
        try:
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as error:
            raise RuntimeError("Install the `vlm` optional dependencies before enabling SmolVLM") from error
        self.model_name = model_name
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.processor = AutoProcessor.from_pretrained(model_name, local_files_only=local_files_only)
        use_half = torch.cuda.is_available() or torch.backends.mps.is_available()
        dtype = torch.float16 if use_half else torch.float32
        self.model = AutoModelForImageTextToText.from_pretrained(model_name, dtype=dtype, local_files_only=local_files_only)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.model.to(self.device).eval()

    @staticmethod
    def _choice(text: str, choices: Sequence[str]) -> Optional[str]:
        lower = text.strip().lower().replace("-", "_").replace(" ", "_")
        for choice in choices:
            if re.search(r"(?:^|[^a-z_]){}(?:$|[^a-z_])".format(re.escape(choice)), lower):
                return choice
        return None

    def _ask(self, image: Image.Image, question: str, max_new_tokens: int = 12) -> str:
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=prompt, images=[image.convert("RGB")], return_tensors="pt").to(self.device)
        with torch.inference_mode():
            generated = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return self.processor.batch_decode(
            generated[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )[0].strip()

    def _predict_one(self, image: Image.Image) -> AffectPrediction:
        responses: Dict[str, str] = {
            "emotion": self._ask(image, EMOTION_PROMPT),
            "cue": self._ask(image, CUE_PROMPT),
            "valence": self._ask(image, VALENCE_PROMPT),
            "arousal": self._ask(image, AROUSAL_PROMPT),
            "confidence": self._ask(image, CONFIDENCE_PROMPT),
            "evidence": self._ask(image, EVIDENCE_PROMPT, max_new_tokens=16),
        }
        emotion = self._choice(responses["emotion"], EMOTIONS)
        cue = self._choice(responses["cue"], ["color_lighting", "facial_action_region", "scene_context", "embedded_text"])
        valence_word = self._choice(responses["valence"], ["negative", "neutral", "positive"])
        arousal_word = self._choice(responses["arousal"], ["low", "medium", "high"])
        confidence_word = self._choice(responses["confidence"], ["low", "medium", "high"])
        valid = all(value is not None for value in [emotion, cue, valence_word, arousal_word, confidence_word])
        emotion = emotion or "surprise"
        cue = cue or "scene_context"
        confidence = {"low": 0.38, "medium": 0.58, "high": 0.78}.get(confidence_word, 1.0 / len(EMOTIONS))
        probabilities = {label: (1.0 - confidence) / (len(EMOTIONS) - 1) for label in EMOTIONS}
        probabilities[emotion] = confidence
        evidence = re.sub(r"\s+", " ", responses["evidence"]).strip()[:240]
        return AffectPrediction(
            emotion_probabilities=probabilities,
            valence={"negative": -0.67, "neutral": 0.0, "positive": 0.67}.get(valence_word, 0.0),
            arousal={"low": -0.50, "medium": 0.0, "high": 0.67}.get(arousal_word, 0.0),
            confidence=confidence,
            predicted_emotion=emotion,
            caption=evidence,
            evidence=evidence,
            evidence_cue=cue,
            raw={"responses": responses, "parse_status": "valid_constrained" if valid else "invalid_constrained"},
        )

    @staticmethod
    def _parse(text: str) -> AffectPrediction:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            lower = text.lower()
            counts = {label: len(re.findall(r"\b{}\b".format(label), lower)) for label in EMOTIONS}
            emotion = max(counts, key=counts.get) if max(counts.values()) else "surprise"
            confidence = 0.35 if max(counts.values()) else 1.0 / len(EMOTIONS)
            probabilities = {label: (1.0 - confidence) / (len(EMOTIONS) - 1) for label in EMOTIONS}
            probabilities[emotion] = confidence
            va_prior = {
                "anger": (-0.65, 0.75), "disgust": (-0.70, 0.45), "fear": (-0.75, 0.85),
                "joy": (0.80, 0.60), "sadness": (-0.75, -0.35), "surprise": (0.10, 0.80),
            }
            if any(word in lower for word in ["face", "smile", "mouth", "eye"]):
                cue = "facial_action_region"
            elif any(word in lower for word in ["text", "word", "sign", "letter"]):
                cue = "embedded_text"
            elif any(word in lower for word in ["colour", "color", "bright", "dark", "light"]):
                cue = "color_lighting"
            else:
                cue = "scene_context"
            cleaned = re.sub(r"\s+", " ", text).strip()
            return AffectPrediction(
                emotion_probabilities=probabilities,
                valence=va_prior[emotion][0],
                arousal=va_prior[emotion][1],
                confidence=confidence,
                predicted_emotion=emotion,
                caption=cleaned[:240],
                evidence=cleaned[:160],
                evidence_cue=cue,
                raw={"response": text, "parse_status": "fallback_free_text"},
            )
        data = json.loads(match.group(0))
        probabilities = {label: max(0.0, float(data.get("emotion_probabilities", {}).get(label, 0.0))) for label in EMOTIONS}
        total = sum(probabilities.values())
        emotion = str(data.get("emotion", "")).lower()
        if total <= 0:
            probabilities = {label: float(label == emotion) for label in EMOTIONS}
        else:
            probabilities = {key: value / total for key, value in probabilities.items()}
        if emotion not in EMOTIONS:
            emotion = max(probabilities, key=probabilities.get)
        return AffectPrediction(
            emotion_probabilities=probabilities,
            valence=max(-1.0, min(1.0, float(data.get("valence", 0.0)))),
            arousal=max(-1.0, min(1.0, float(data.get("arousal", 0.0)))),
            confidence=max(0.0, min(1.0, float(data.get("confidence", probabilities[emotion])))),
            predicted_emotion=emotion,
            caption=str(data.get("caption", "")),
            evidence=str(data.get("evidence", "")),
            evidence_cue=str(data.get("evidence_cue", "")),
            raw={"response": text, "parse_status": "valid_json"},
        )

    def _cache_path(self, key: str) -> Optional[Path]:
        if not self.cache_dir:
            return None
        digest = hashlib.sha256(
            (self.model_name + "|constrained-v2|" + key).encode("utf-8")
        ).hexdigest()
        return self.cache_dir / "{}.json".format(digest)

    def predict(self, images: List[Image.Image], cache_keys: Optional[Sequence[str]] = None) -> List[AffectPrediction]:
        if cache_keys is not None and len(cache_keys) != len(images):
            raise ValueError("cache_keys must align with images")
        predictions = []
        for index, image in enumerate(images):
            key = cache_keys[index] if cache_keys is not None else "image-{}".format(index)
            cache_path = self._cache_path(key)
            if cache_path and cache_path.is_file():
                predictions.append(AffectPrediction(**json.loads(cache_path.read_text())))
                continue
            prediction = self._predict_one(image)
            if cache_path:
                temporary = cache_path.with_suffix(".tmp")
                temporary.write_text(json.dumps(prediction.to_dict(), indent=2, sort_keys=True))
                temporary.replace(cache_path)
            predictions.append(prediction)
        return predictions
