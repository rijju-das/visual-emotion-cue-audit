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
        superpixel_fallback: bool = False,
        superpixel_count: int = 48,
        panoptic_model: str = "facebook/mask2former-swin-small-coco-panoptic",
        panoptic_score_threshold: float = 0.55,
        panoptic_min_area_fraction: float = 0.03,
        panoptic_local_files_only: bool = False,
        panoptic_provider: Optional[Callable[[Image.Image], List[Dict]]] = None,
        semantic_model: str = "nvidia/segformer-b0-finetuned-ade-512-512",
        semantic_score_threshold: float = 0.45,
        semantic_local_files_only: bool = False,
        semantic_provider: Optional[Callable[[Image.Image], List[Dict]]] = None,
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
        self.panoptic_model_name = panoptic_model
        self.panoptic_score_threshold = panoptic_score_threshold
        self.panoptic_min_area_fraction = panoptic_min_area_fraction
        self.panoptic_local_files_only = panoptic_local_files_only
        self.panoptic_provider = panoptic_provider
        self.semantic_model_name = semantic_model
        self.semantic_score_threshold = semantic_score_threshold
        self.semantic_local_files_only = semantic_local_files_only
        self.semantic_provider = semantic_provider
        self.mask_provider = mask_provider
        self._model = None
        self._categories = None
        self._device = None
        self._semantic_processor = None
        self._semantic_model = None
        self._semantic_device = None
        self._panoptic_processor = None
        self._panoptic_model = None
        self._panoptic_device = None
        self.unavailable_reason = ""

    @staticmethod
    def _label_priority(label: str) -> int:
        label = label.lower()
        ordered = ["person", "wall", "floor", "ceiling", "door", "window", "building", "road", "sky", "earth", "grass"]
        return next((index for index, name in enumerate(ordered) if name in label), 50)

    def _initialise_panoptic(self) -> bool:
        if self.panoptic_provider is not None or self._panoptic_model is not None:
            return True
        try:
            import torch
            from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

            self._panoptic_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._panoptic_processor = AutoImageProcessor.from_pretrained(
                self.panoptic_model_name,
                local_files_only=self.panoptic_local_files_only,
            )
            self._panoptic_model = Mask2FormerForUniversalSegmentation.from_pretrained(
                self.panoptic_model_name,
                local_files_only=self.panoptic_local_files_only,
            ).to(self._panoptic_device).eval()
            return True
        except (ImportError, OSError, RuntimeError, ValueError) as error:
            self.unavailable_reason = "panoptic_segmenter_unavailable:{}".format(type(error).__name__)
            return False

    def _panoptic_regions(self, image: Image.Image) -> List[Dict]:
        if self.panoptic_provider is not None:
            raw_regions = self.panoptic_provider(image)
        else:
            if not self._initialise_panoptic():
                return []
            import torch

            inputs = self._panoptic_processor(images=image.convert("RGB"), return_tensors="pt")
            inputs = {name: value.to(self._panoptic_device) for name, value in inputs.items()}
            with torch.inference_mode():
                outputs = self._panoptic_model(**inputs)
            processed = self._panoptic_processor.post_process_panoptic_segmentation(
                outputs,
                threshold=self.panoptic_score_threshold,
                mask_threshold=0.5,
                overlap_mask_area_threshold=0.8,
                label_ids_to_fuse=set(),
                target_sizes=[(image.height, image.width)],
            )[0]
            segmentation = processed["segmentation"].detach().cpu().numpy()
            raw_regions = []
            for segment in processed["segments_info"]:
                label_id = int(segment["label_id"])
                raw_regions.append({
                    "mask": segmentation == int(segment["id"]),
                    "label": str(self._panoptic_model.config.id2label.get(label_id, label_id)),
                    "score": float(segment.get("score", 1.0)),
                    "segment_id": int(segment["id"]),
                    "label_id": label_id,
                    "was_fused": bool(segment.get("was_fused", False)),
                })

        regions = []
        for region_index, region in enumerate(raw_regions):
            raw_mask = region["mask"]
            if isinstance(raw_mask, Image.Image):
                mask = np.asarray(raw_mask.resize(image.size, Image.Resampling.NEAREST)) > 0
            else:
                mask = np.asarray(raw_mask).squeeze().astype(bool)
                if mask.shape != (image.height, image.width):
                    mask = np.asarray(
                        Image.fromarray(mask).resize(image.size, Image.Resampling.NEAREST)
                    ).astype(bool)
            area = float(mask.mean())
            score = float(region.get("score", 1.0))
            if score < self.panoptic_score_threshold or not self.panoptic_min_area_fraction <= area <= 0.90:
                continue
            label = str(region.get("label", "object")).strip().lower()
            label_id = int(region.get("label_id", -1))
            is_thing = 0 <= label_id < 80
            priority = self._label_priority(label)
            if is_thing and priority == 50:
                priority = 12
            regions.append({
                "binary": mask,
                "label": label,
                "score": score,
                "area_fraction": area,
                "segmenter": "mask2former_coco_panoptic",
                "priority": priority,
                "segment_id": int(region.get("segment_id", region_index)),
                "label_id": label_id,
                "was_fused": bool(region.get("was_fused", False)),
                "derived_region": False,
                "is_thing": is_thing,
            })
        person_union = np.zeros((image.height, image.width), dtype=bool)
        for region in regions:
            if "person" in region["label"]:
                person_union |= region["binary"]
        luminance = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
        dark_threshold = min(0.30, float(np.quantile(luminance, 0.28)))
        dark_background = (luminance <= dark_threshold) & ~person_union
        dark_area = float(dark_background.mean())
        if self.panoptic_min_area_fraction <= dark_area <= 0.65:
            regions.append({
                "binary": dark_background,
                "label": "dark-background",
                "score": 1.0,
                "area_fraction": dark_area,
                "segmenter": "mask2former_panoptic_plus_luminance",
                "priority": 11,
                "segment_id": -1,
                "label_id": -1,
                "was_fused": False,
                "derived_region": True,
                "is_thing": False,
            })
        regions.sort(key=lambda region: (region["priority"], -region["area_fraction"]))
        if regions:
            self.unavailable_reason = ""
        return regions[: self.max_candidates]

    def _initialise_semantic(self) -> bool:
        if self.semantic_provider is not None or self._semantic_model is not None:
            return True
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

            self._semantic_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._semantic_processor = AutoImageProcessor.from_pretrained(
                self.semantic_model_name,
                local_files_only=self.semantic_local_files_only,
            )
            self._semantic_model = AutoModelForSemanticSegmentation.from_pretrained(
                self.semantic_model_name,
                local_files_only=self.semantic_local_files_only,
            ).to(self._semantic_device).eval()
            return True
        except (ImportError, OSError, RuntimeError, ValueError) as error:
            self.unavailable_reason = "semantic_segmenter_unavailable:{}".format(type(error).__name__)
            return False

    def _semantic_regions(self, image: Image.Image) -> List[Dict]:
        if self.semantic_provider is not None:
            raw_regions = self.semantic_provider(image)
        else:
            if not self._initialise_semantic():
                return []
            import torch

            inputs = self._semantic_processor(images=image.convert("RGB"), return_tensors="pt")
            inputs = {name: value.to(self._semantic_device) for name, value in inputs.items()}
            with torch.inference_mode():
                outputs = self._semantic_model(**inputs)
                probabilities = torch.nn.functional.interpolate(
                    outputs.logits,
                    size=(image.height, image.width),
                    mode="bilinear",
                    align_corners=False,
                ).softmax(dim=1)[0]
                confidence, segmentation = probabilities.max(dim=0)
            segmentation = segmentation.detach().cpu().numpy()
            confidence = confidence.detach().cpu().numpy()
            raw_regions = []
            for class_id in np.unique(segmentation):
                mask = segmentation == class_id
                label = self._semantic_model.config.id2label.get(
                    int(class_id), str(int(class_id))
                )
                raw_regions.append({
                    "mask": mask,
                    "label": str(label).strip().lower(),
                    "score": float(confidence[mask].mean()),
                })

        priority = {
            "person": 0,
            "wall": 1,
            "floor": 2,
            "ceiling": 3,
            "door": 4,
            "windowpane": 5,
            "building": 6,
            "road": 7,
            "sky": 8,
            "earth": 9,
            "grass": 10,
        }
        regions = []
        person_union = np.zeros((image.height, image.width), dtype=bool)
        for region in raw_regions:
            raw_mask = region["mask"]
            if isinstance(raw_mask, Image.Image):
                mask = np.asarray(raw_mask.resize(image.size, Image.Resampling.NEAREST)) > 0
            else:
                mask = np.asarray(raw_mask).squeeze().astype(bool)
                if mask.shape != (image.height, image.width):
                    mask = np.asarray(
                        Image.fromarray(mask).resize(image.size, Image.Resampling.NEAREST)
                    ).astype(bool)
            area = float(mask.mean())
            score = float(region.get("score", 1.0))
            label = str(region.get("label", "semantic_region")).strip().lower()
            if score < self.semantic_score_threshold:
                continue
            if not self.min_area_fraction <= area <= min(0.85, self.max_area_fraction + 0.20):
                continue
            if label == "person":
                person_union |= mask
            regions.append({
                "binary": mask,
                "label": label,
                "score": score,
                "area_fraction": area,
                "segmenter": "segformer_ade20k_semantic",
                "priority": priority.get(label.split(",")[0], 50),
            })

        # A perceptually dark background is useful even when ADE20K divides it
        # across wall, floor, and other stuff classes. Person pixels are excluded.
        luminance = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
        dark_threshold = min(0.30, float(np.quantile(luminance, 0.28)))
        dark_background = (luminance <= dark_threshold) & ~person_union
        dark_area = float(dark_background.mean())
        if self.min_area_fraction <= dark_area <= 0.65:
            regions.append({
                "binary": dark_background,
                "label": "dark_background",
                "score": 1.0,
                "area_fraction": dark_area,
                "segmenter": "segformer_plus_luminance_background",
                "priority": 11,
            })
        regions.sort(key=lambda region: (region["priority"], -region["area_fraction"]))
        if regions:
            self.unavailable_reason = ""
        return regions[: self.max_candidates]

    def _semantic_twins(self, image: Image.Image, regions: List[Dict]) -> List[GeneratedTwin]:
        twins = []
        for region_index, region in enumerate(regions):
            mask = Image.fromarray(region["binary"].astype(np.uint8) * 255, mode="L")
            label = region["label"]
            twins.append(GeneratedTwin(
                image=luminance_preserving_desaturate(image, mask),
                mask=mask,
                cue_family=self.cue_family,
                operation="semantic_region_chroma_removal",
                target_region=mask_bbox(mask),
                metadata={
                    "intervention_scope": "semantic_scene_entity",
                    "semantic_region_id": region_index,
                    "semantic_label": label,
                    "semantic_score": region["score"],
                    "semantic_mask_area_fraction": region["area_fraction"],
                    "semantic_segmenter": region["segmenter"],
                    "is_complete_person_region": label == "person",
                    "is_control": False,
                },
            ))
        return twins

    def _panoptic_twins(self, image: Image.Image, regions: List[Dict]) -> List[GeneratedTwin]:
        twins = []
        for region_index, region in enumerate(regions):
            mask = Image.fromarray(region["binary"].astype(np.uint8) * 255, mode="L")
            label = region["label"]
            twins.append(GeneratedTwin(
                image=luminance_preserving_desaturate(image, mask),
                mask=mask,
                cue_family=self.cue_family,
                operation="panoptic_entity_chroma_removal",
                target_region=mask_bbox(mask),
                metadata={
                    "intervention_scope": "complete_panoptic_entity",
                    "panoptic_region_index": region_index,
                    "panoptic_segment_id": region["segment_id"],
                    "panoptic_label_id": region["label_id"],
                    "panoptic_label": label,
                    "panoptic_score": region["score"],
                    "panoptic_mask_area_fraction": region["area_fraction"],
                    "panoptic_segmenter": region["segmenter"],
                    "panoptic_was_fused": region["was_fused"],
                    "derived_region": region.get("derived_region", False),
                    "panoptic_is_thing": region.get("is_thing", False),
                    "is_complete_person_region": label == "person",
                    "mask_rectangularity": self._rectangularity(region["binary"]),
                    "is_control": False,
                },
            ))
        return twins

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
        unique_labels = [int(label) for label in np.unique(labels)]
        global_colour = rgb.mean(axis=(0, 1))
        mean_colours = {}
        saliencies = {}
        adjacency = {label: set() for label in unique_labels}
        for label in unique_labels:
            small_region = labels == label
            mean_colours[label] = rgb[small_region].mean(axis=0)
            colour_contrast = float(np.linalg.norm(mean_colours[label] - global_colour))
            centrality = float(centre_prior[small_region].mean())
            touches_border = bool(
                small_region[0].any() or small_region[-1].any()
                or small_region[:, 0].any() or small_region[:, -1].any()
            )
            saliencies[label] = 0.55 * centrality + 0.45 * min(1.0, colour_contrast * 2.0)
            if touches_border:
                saliencies[label] *= 0.78

        for left, right in zip(labels[:, :-1].ravel(), labels[:, 1:].ravel()):
            if left != right:
                adjacency[int(left)].add(int(right))
                adjacency[int(right)].add(int(left))
        for top, bottom in zip(labels[:-1].ravel(), labels[1:].ravel()):
            if top != bottom:
                adjacency[int(top)].add(int(bottom))
                adjacency[int(bottom)].add(int(top))

        proposals = []
        for seed in sorted(unique_labels, key=lambda label: saliencies[label], reverse=True):
            grown = {seed}
            frontier = set(adjacency[seed])
            while frontier and len(grown) < 12:
                neighbour = min(
                    frontier,
                    key=lambda label: float(np.linalg.norm(mean_colours[label] - mean_colours[seed])),
                )
                frontier.remove(neighbour)
                if float(np.linalg.norm(mean_colours[neighbour] - mean_colours[seed])) > 0.20:
                    continue
                candidate_labels = grown | {neighbour}
                candidate_area = float(np.isin(full_labels, list(candidate_labels)).mean())
                if candidate_area > self.max_area_fraction:
                    continue
                grown.add(neighbour)
                frontier.update(adjacency[neighbour] - grown)

            # One SLIC cell can retain its rectangular initialization. Requiring
            # multiple connected cells and rejecting high rectangular fill makes
            # a patch-shaped fallback impossible.
            if len(grown) < 2:
                continue
            full_region = np.isin(full_labels, list(grown))
            area = float(full_region.mean())
            rectangularity = self._rectangularity(full_region)
            if not self.min_area_fraction <= area <= self.max_area_fraction or rectangularity >= 0.86:
                continue
            proposals.append({
                "mask": full_region.astype(np.float32),
                "label": "superpixel_component_{}".format(seed),
                "score": max(0.01, saliencies[seed]),
                "segmenter": "numpy_slic_superpixel_connected_region_fallback",
                "component_count": len(grown),
                "mask_rectangularity": rectangularity,
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

    @staticmethod
    def _rectangularity(mask: np.ndarray) -> float:
        rows, columns = np.where(mask)
        if not len(rows):
            return 1.0
        bounding_area = (rows.max() - rows.min() + 1) * (columns.max() - columns.min() + 1)
        return float(mask.sum() / bounding_area)

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
            rectangularity = self._rectangularity(binary)
            segmenter = str(proposal.get("segmenter", "provided_object_masks"))
            maximum_rectangularity = 0.86 if "superpixel" in segmenter else 0.97
            if rectangularity >= maximum_rectangularity:
                continue
            if any(self._iou(binary, item["binary"]) >= 0.85 for item in accepted):
                continue
            accepted.append({
                "binary": binary,
                "label": str(proposal.get("label", "object")),
                "score": float(proposal.get("score", 1.0)),
                "segmenter": segmenter,
                "mask_rectangularity": rectangularity,
                "superpixel_component_count": int(proposal.get("component_count", 0)),
                "proposal_index": proposal_index,
                "area_fraction": area_fraction,
            })
            if len(accepted) >= self.max_candidates:
                break
        return accepted

    def generate(self, image: Image.Image) -> List[GeneratedTwin]:
        # Normal runs first retain every detected panoptic object/surface as a
        # separate counterfactual candidate. Injected providers isolate unit tests.
        skip_panoptic = (
            self.panoptic_provider is None
            and (self.mask_provider is not None or self.semantic_provider is not None)
        )
        panoptic_regions = [] if skip_panoptic else self._panoptic_regions(image)
        if panoptic_regions:
            return self._panoptic_twins(image, panoptic_regions)

        semantic_regions = [] if self.mask_provider is not None and self.semantic_provider is None else self._semantic_regions(image)
        if semantic_regions:
            return self._semantic_twins(image, semantic_regions)

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
        subject_rectangularity = self._rectangularity(subject_binary)
        shared = {
            "subject_type": subject_type,
            "subject_labels": [proposal["label"] for proposal in subject_proposals],
            "subject_instance_count": len(subject_proposals),
            "subject_mask_area_fraction": float(subject_binary.mean()),
            "subject_mask_rectangularity": subject_rectangularity,
            "superpixel_component_count": int(sum(
                proposal.get("superpixel_component_count", 0) for proposal in subject_proposals
            )),
            "object_segmenter": ";".join(segmenters),
            "used_superpixel_fallback": any(
                "superpixel" in name or "slic" in name for name in segmenters
            ),
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
