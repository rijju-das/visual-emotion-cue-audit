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
        superpixel_fallback: bool = True,
        superpixel_count: int = 48,
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
        self.superpixel_fallback = superpixel_fallback
        self.superpixel_count = superpixel_count
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
            proposals = self.mask_provider(image)
            return proposals or self._superpixel_proposals(image)
        if not self._initialise():
            return self._superpixel_proposals(image)

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
                "segmenter": "torchvision_maskrcnn_resnet50_fpn_v2",
            })
        if proposals:
            return proposals
        return self._superpixel_proposals(image)

    def _superpixel_proposals(self, image: Image.Image) -> List[Dict]:
        """Create non-rectangular SLIC-style regions without an extra dependency."""
        if not self.superpixel_fallback:
            return []
        width, height = image.size
        scale = min(1.0, 192.0 / max(width, height))
        work_width = max(16, int(round(width * scale)))
        work_height = max(16, int(round(height * scale)))
        rgb = np.asarray(
            image.convert("RGB").resize((work_width, work_height), Image.Resampling.BILINEAR),
            dtype=np.float32,
        ) / 255.0
        count = max(4, min(self.superpixel_count, work_width * work_height // 64))
        step = max(2.0, np.sqrt(work_width * work_height / count))
        centres = []
        for y in np.arange(step / 2.0, work_height, step):
            for x in np.arange(step / 2.0, work_width, step):
                yi, xi = min(work_height - 1, int(round(y))), min(work_width - 1, int(round(x)))
                centres.append([y, x, *rgb[yi, xi]])
        centres = np.asarray(centres, dtype=np.float32)
        labels = np.full((work_height, work_width), -1, dtype=np.int32)
        yy, xx = np.mgrid[0:work_height, 0:work_width]
        for _ in range(5):
            distances = np.full((work_height, work_width), np.inf, dtype=np.float32)
            for index, centre in enumerate(centres):
                cy, cx = centre[:2]
                y0, y1 = max(0, int(cy - 2 * step)), min(work_height, int(cy + 2 * step + 1))
                x0, x1 = max(0, int(cx - 2 * step)), min(work_width, int(cx + 2 * step + 1))
                colour_distance = np.square(rgb[y0:y1, x0:x1] - centre[2:]).sum(axis=2)
                spatial_distance = (
                    np.square((yy[y0:y1, x0:x1] - cy) / step)
                    + np.square((xx[y0:y1, x0:x1] - cx) / step)
                )
                distance = colour_distance + 0.08 * spatial_distance
                update = distance < distances[y0:y1, x0:x1]
                distances[y0:y1, x0:x1][update] = distance[update]
                labels[y0:y1, x0:x1][update] = index
            for index in range(len(centres)):
                region = labels == index
                if region.any():
                    centres[index, 0] = yy[region].mean()
                    centres[index, 1] = xx[region].mean()
                    centres[index, 2:] = rgb[region].mean(axis=0)

        full_labels = np.asarray(
            Image.fromarray(labels).resize(image.size, Image.Resampling.NEAREST), dtype=np.int32
        )
        centre_prior = np.exp(
            -(
                np.square((xx - (work_width - 1) / 2.0) / max(1.0, work_width * 0.42))
                + np.square((yy - (work_height - 1) / 2.0) / max(1.0, work_height * 0.42))
            )
        )
        global_colour = rgb.mean(axis=(0, 1))
        proposals = []
        for label in np.unique(labels):
            small_region = labels == label
            full_region = full_labels == label
            area = float(full_region.mean())
            if not self.min_area_fraction <= area <= self.max_area_fraction:
                continue
            colour_contrast = float(np.linalg.norm(rgb[small_region].mean(axis=0) - global_colour))
            centrality = float(centre_prior[small_region].mean())
            touches_border = bool(
                small_region[0].any() or small_region[-1].any() or small_region[:, 0].any() or small_region[:, -1].any()
            )
            saliency = 0.55 * centrality + 0.45 * min(1.0, colour_contrast * 2.0)
            if touches_border:
                saliency *= 0.78
            proposals.append({
                "mask": full_region.astype(np.float32),
                "label": "superpixel_region_{}".format(int(label)),
                "score": max(0.01, saliency),
                "segmenter": "numpy_slic_superpixel_fallback",
            })
        proposals.sort(key=lambda proposal: proposal["score"], reverse=True)
        if proposals:
            self.unavailable_reason = ""
        return proposals[: self.max_candidates]

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
                "segmenter": str(proposal.get("segmenter", "provided_object_masks")),
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
            # For non-person scenes, use one complete dominant object or, when
            # detection failed, the most salient boundary-following superpixel.
            primary = max(proposals, key=lambda proposal: proposal["area_fraction"] * proposal["score"])
            subject_binary = primary["binary"]
            subject_proposals = [primary]
            subject_type = (
                "salient_superpixel_region"
                if "superpixel" in primary["segmenter"]
                else "dominant_detected_object"
            )

        subject_mask = Image.fromarray(subject_binary.astype(np.uint8) * 255, mode="L")
        background_mask = Image.fromarray((~subject_binary).astype(np.uint8) * 255, mode="L")
        segmenters = sorted({proposal["segmenter"] for proposal in subject_proposals})
        shared = {
            "subject_type": subject_type,
            "subject_labels": [proposal["label"] for proposal in subject_proposals],
            "subject_instance_count": len(subject_proposals),
            "subject_mask_area_fraction": float(subject_binary.mean()),
            "object_segmenter": ";".join(segmenters),
            "used_superpixel_fallback": any("superpixel" in name for name in segmenters),
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
