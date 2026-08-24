"""Object-localised, luminance-preserving colour interventions."""

from typing import Callable, Dict, List, Optional

import numpy as np
from PIL import Image

from ..schema import CueFamily
from .base import GeneratedTwin, attenuate_illumination, luminance_preserving_desaturate, mask_bbox


class ColorIntervention:
    """Remove chroma from complete detected objects rather than image patches."""

    cue_family = CueFamily.COLOR

    def __init__(
        self,
        grid_size: int = 4,
        backend: str = "maskrcnn",
        score_threshold: float = 0.65,
        mask_threshold: float = 0.5,
        min_area_fraction: float = 0.01,
        max_area_fraction: float = 0.65,
        max_candidates: int = 8,
        mask_provider: Optional[Callable[[Image.Image], List[Dict]]] = None,
    ):
        # Retained for compatibility with old configs; grid cells are no longer used.
        self.grid_size = grid_size
        self.backend = backend
        self.score_threshold = score_threshold
        self.mask_threshold = mask_threshold
        self.min_area_fraction = min_area_fraction
        self.max_area_fraction = max_area_fraction
        self.max_candidates = max_candidates
        self.mask_provider = mask_provider
        self._model = None
        self._categories = None
        self._device = None
        self.unavailable_reason = ""

    def _initialise(self) -> bool:
        if self.mask_provider is not None or self._model is not None:
            return True
        if self.backend != "maskrcnn":
            self.unavailable_reason = "unsupported_object_mask_backend:{}".format(self.backend)
            return False
        try:
            import torch
            from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights, maskrcnn_resnet50_fpn_v2

            weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._model = maskrcnn_resnet50_fpn_v2(weights=weights).to(self._device).eval()
            self._categories = weights.meta["categories"]
            return True
        except (ImportError, OSError, RuntimeError) as error:
            self.unavailable_reason = "object_segmenter_unavailable:{}".format(type(error).__name__)
            return False

    def _detect(self, image: Image.Image) -> List[Dict]:
        if self.mask_provider is not None:
            return self.mask_provider(image)
        if not self._initialise():
            return []

        import torch
        from torchvision.transforms.functional import pil_to_tensor

        tensor = pil_to_tensor(image.convert("RGB")).float().div(255.0).to(self._device)
        with torch.inference_mode():
            prediction = self._model([tensor])[0]
        proposals = []
        for mask, label, score in zip(prediction["masks"], prediction["labels"], prediction["scores"]):
            confidence = float(score.item())
            if confidence < self.score_threshold:
                continue
            label_index = int(label.item())
            proposals.append({
                "mask": mask[0].detach().cpu().numpy(),
                "label": self._categories[label_index],
                "score": confidence,
            })
        return proposals

    @staticmethod
    def _iou(left: np.ndarray, right: np.ndarray) -> float:
        intersection = np.logical_and(left, right).sum()
        union = np.logical_or(left, right).sum()
        return float(intersection / union) if union else 0.0

    def _object_masks(self, image: Image.Image) -> List[Dict]:
        width, height = image.size
        accepted = []
        for proposal_index, proposal in enumerate(self._detect(image)):
            raw = proposal["mask"]
            if isinstance(raw, Image.Image):
                raw = np.asarray(raw.resize(image.size, Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
            else:
                raw = np.asarray(raw, dtype=np.float32).squeeze()
                if raw.shape != (height, width):
                    raw = np.asarray(
                        Image.fromarray(raw).resize(image.size, Image.Resampling.BILINEAR), dtype=np.float32
                    )
            binary = raw >= self.mask_threshold
            area_fraction = float(binary.mean())
            if not self.min_area_fraction <= area_fraction <= self.max_area_fraction:
                continue
            if any(self._iou(binary, item["binary"]) >= 0.85 for item in accepted):
                continue
            accepted.append({
                "binary": binary,
                "label": str(proposal.get("label", "object")),
                "score": float(proposal.get("score", 1.0)),
                "proposal_index": proposal_index,
                "area_fraction": area_fraction,
            })
            if len(accepted) >= self.max_candidates:
                break
        return accepted

    def generate(self, image: Image.Image) -> List[GeneratedTwin]:
        proposals = self._object_masks(image)
        if not proposals:
            if not self.unavailable_reason:
                self.unavailable_reason = "no_object_instance_detected"
            return []

        people = [proposal for proposal in proposals if proposal["label"].lower() == "person"]
        if people:
            # Merge every detected person instance so no person is reduced to a face
            # or torso patch and visible arms, hands, legs, and feet remain included.
            subject_binary = np.logical_or.reduce([proposal["binary"] for proposal in people])
            subject_proposals = people
            subject_type = "all_detected_people"
        else:
            # For non-person scenes, use one complete dominant object rather than a grid cell.
            primary = max(proposals, key=lambda proposal: proposal["area_fraction"] * proposal["score"])
            subject_binary = primary["binary"]
            subject_proposals = [primary]
            subject_type = "dominant_detected_object"

        subject_mask = Image.fromarray(subject_binary.astype(np.uint8) * 255, mode="L")
        background_mask = Image.fromarray((~subject_binary).astype(np.uint8) * 255, mode="L")
        segmenter = (
            "torchvision_maskrcnn_resnet50_fpn_v2"
            if self.mask_provider is None
            else "provided_object_masks"
        )
        shared = {
            "subject_type": subject_type,
            "subject_labels": [proposal["label"] for proposal in subject_proposals],
            "subject_instance_count": len(subject_proposals),
            "subject_mask_area_fraction": float(subject_binary.mean()),
            "object_segmenter": segmenter,
            "selection_rule": "merge_all_people_else_largest_area_times_confidence",
            "is_control": False,
        }
        twins = [
            GeneratedTwin(
                image=luminance_preserving_desaturate(image, subject_mask),
                mask=subject_mask,
                cue_family=self.cue_family,
                operation="complete_subject_chroma_removal",
                target_region=mask_bbox(subject_mask),
                metadata={**shared, "intervention_scope": "complete_subject"},
            ),
            GeneratedTwin(
                image=attenuate_illumination(image, background_mask, exposure=0.65),
                mask=background_mask,
                cue_family=self.cue_family,
                operation="background_exposure_reduction",
                target_region=mask_bbox(background_mask),
                metadata={
                    **shared,
                    "intervention_scope": "background_illumination",
                    "exposure_multiplier_linear_rgb": 0.65,
                },
            ),
        ]
        if not twins and not self.unavailable_reason:
            self.unavailable_reason = "no_object_instance_detected"
        return twins
