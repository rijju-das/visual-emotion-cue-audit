"""Typed contracts shared by datasets, interventions, models, and metrics."""

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


EMOTIONS = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]
EMOTION6_ANNOTATION_LABELS = EMOTIONS + ["neutral"]


class CueFamily(str, Enum):
    COLOR = "color_lighting"
    FACE = "facial_action_region"
    CONTEXT = "scene_context"
    TEXT = "embedded_text"


@dataclass
class AffectSample:
    sample_id: str
    image_path: str
    emotion: str
    emotion_distribution: Dict[str, float] = field(default_factory=dict)
    valence: Optional[float] = None
    arousal: Optional[float] = None
    nominal_emotion: Optional[str] = None
    human_plurality_emotion: Optional[str] = None
    human_plurality_probability: Optional[float] = None
    split: str = "unspecified"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Intervention:
    intervention_id: str
    sample_id: str
    cue_family: CueFamily
    operation: str
    image_path: str
    mask_path: str
    eligible: bool = True
    skip_reason: str = ""
    target_region: Optional[Tuple[int, int, int, int]] = None
    expected_direction: str = "attenuate_source_affect"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["cue_family"] = self.cue_family.value
        return result


@dataclass
class AffectPrediction:
    emotion_probabilities: Dict[str, float]
    valence: float
    arousal: float
    confidence: float
    predicted_emotion: str
    caption: str = ""
    evidence: str = ""
    evidence_cue: str = ""
    raw: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
