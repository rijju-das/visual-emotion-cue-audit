"""MediaPipe-localised facial action-region evidence ablations."""

import platform
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from ..schema import CueFamily
from .base import GeneratedTwin, mask_bbox


LANDMARK_COMPONENTS: Dict[str, Sequence[Sequence[int]]] = {
    # Separate polygons preserve the two eyebrows and two eyes instead of
    # replacing the entire horizontal facial band between them.
    "brow_AU1_2_4": [
        [70, 63, 105, 66, 107, 55, 65, 52, 53, 46],
        [336, 296, 334, 293, 300, 285, 295, 282, 283, 276],
    ],
    "eye_AU5_6_7": [
        [33, 160, 158, 133, 153, 144],
        [362, 385, 387, 263, 373, 380],
    ],
    "mouth_AU10_12_15_20_23_24_25_26": [
        [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95],
    ],
}

AU_GROUPS: Dict[str, Sequence[str]] = {
    "brow_AU1_2_4": ["AU1", "AU2", "AU4"],
    "eye_AU5_6_7": ["AU5", "AU6", "AU7"],
    "mouth_AU10_12_15_20_23_24_25_26": ["AU10", "AU12", "AU15", "AU20", "AU23", "AU24", "AU25", "AU26"],
}


class FaceActionRegionIntervention:
    cue_family = CueFamily.FACE

    def __init__(self, model_path: str, backend: str = "auto", dnn_prototxt: str = "", dnn_model: str = ""):
        self.model_path = Path(model_path)
        self.backend = "opencv" if backend == "auto" and platform.system() == "Darwin" else "mediapipe" if backend == "auto" else backend
        self.dnn_prototxt = Path(dnn_prototxt) if dnn_prototxt else None
        self.dnn_model = Path(dnn_model) if dnn_model else None
        self._opencv_net = None
        self._landmarker = None
        self.unavailable_reason = ""

    def _initialise(self) -> bool:
        if self._landmarker is not None:
            return True
        if not self.model_path.exists():
            self.unavailable_reason = "face_landmarker_model_missing"
            return False
        try:
            import mediapipe as mp

            options = mp.tasks.vision.FaceLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(
                    model_asset_path=str(self.model_path),
                    delegate=mp.tasks.BaseOptions.Delegate.CPU,
                ),
                running_mode=mp.tasks.vision.RunningMode.IMAGE,
                num_faces=3,
            )
            self._landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)
            return True
        except (ImportError, AttributeError, RuntimeError) as error:
            self.unavailable_reason = "mediapipe_unavailable:{}".format(type(error).__name__)
            return False

    def _detect(self, image: Image.Image) -> List[List[Tuple[float, float]]]:
        if not self._initialise():
            return []
        import mediapipe as mp

        array = np.asarray(image.convert("RGB"), dtype=np.uint8)
        result = self._landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=array))
        return [[(point.x, point.y) for point in face] for face in result.face_landmarks]

    def _opencv_regions(self, image: Image.Image, require_person_support: bool = True):
        try:
            import cv2
        except ImportError:
            self.unavailable_reason = "opencv_unavailable"
            return []
        array = np.asarray(image.convert("RGB"), dtype=np.uint8)
        if not self.dnn_prototxt or not self.dnn_model or not self.dnn_prototxt.exists() or not self.dnn_model.exists():
            self.unavailable_reason = "opencv_dnn_face_assets_missing"
            return []
        if self._opencv_net is None:
            self._opencv_net = cv2.dnn.readNetFromCaffe(str(self.dnn_prototxt), str(self.dnn_model))
        bgr = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
        blob = cv2.dnn.blobFromImage(cv2.resize(bgr, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        self._opencv_net.setInput(blob)
        detections = self._opencv_net.forward()
        height_image, width_image = array.shape[:2]
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        people, _ = hog.detectMultiScale(bgr, winStride=(4, 4), padding=(8, 8), scale=1.03)
        faces = []
        for index in range(detections.shape[2]):
            confidence = float(detections[0, 0, index, 2])
            if confidence < 0.78:
                continue
            left, top, right, bottom = detections[0, 0, index, 3:7] * np.asarray([width_image, height_image, width_image, height_image])
            x, y = max(0, int(left)), max(0, int(top))
            right, bottom = min(width_image, int(right)), min(height_image, int(bottom))
            face_width, face_height = right - x, bottom - y
            human_supported = not require_person_support or any(
                max(x, person_x) < min(right, person_x + person_width)
                and person_y <= bottom + 0.45 * face_height
                and person_y + person_height > y
                for person_x, person_y, person_width, person_height in people
            )
            if face_width >= 36 and face_height >= 36 and human_supported:
                faces.append((x, y, right - x, bottom - y, confidence))
        regions = []
        fractions = {
            "brow_AU1_2_4": (0.12, 0.20, 0.88, 0.43),
            "eye_AU5_6_7": (0.08, 0.30, 0.92, 0.58),
            "mouth_AU10_12_15_20_23_24_25_26": (0.18, 0.58, 0.82, 0.92),
        }
        for face_index, (x, y, width, height, confidence) in enumerate(faces):
            for group_name, (left, top, right, bottom) in fractions.items():
                box = (x + left * width, y + top * height, x + right * width, y + bottom * height)
                mask = Image.new("L", image.size, 0)
                ImageDraw.Draw(mask).ellipse(box, fill=255)
                mask = mask.filter(ImageFilter.GaussianBlur(max(2.0, min(image.size) / 160.0)))
                regions.append((face_index, group_name, mask, "OpenCV approximate anatomical fallback", confidence, 1))
        return regions

    @staticmethod
    def _group_mask(
        image: Image.Image,
        face: List[Tuple[float, float]],
        components: Sequence[Sequence[int]],
        group_name: str = "",
    ) -> Image.Image:
        width, height = image.size
        mask = Image.new("L", image.size, 0)
        draw = ImageDraw.Draw(mask)
        for indices in components:
            points = [(face[index][0] * width, face[index][1] * height) for index in indices]
            draw.polygon(points, fill=255)
        face_width = (max(point[0] for point in face) - min(point[0] for point in face)) * width
        dilation_fraction = (
            0.055 if group_name.startswith("mouth")
            else 0.045 if group_name.startswith("brow")
            else 0.040
        )
        dilation_radius = max(2, int(round(face_width * dilation_fraction)))
        dilation = 2 * dilation_radius + 1
        mask = mask.filter(ImageFilter.MaxFilter(dilation))
        return mask.filter(ImageFilter.GaussianBlur(max(1.0, dilation_radius / 3.0)))

    @staticmethod
    def _strong_ablation(image: Image.Image, mask: Image.Image) -> Image.Image:
        bbox = mask_bbox(mask)
        if bbox is None:
            return image.copy()
        span = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
        blur_radius = max(7.0, span * 0.22)
        blurred = image.convert("RGB").filter(ImageFilter.GaussianBlur(blur_radius))
        # Pixelation suppresses residual eye/lip/brow geometry that Gaussian blur
        # alone can preserve in low-resolution face crops.
        width, height = image.size
        block = max(6, int(round(span / 8.0)))
        pixelated = blurred.resize(
            (max(1, width // block), max(1, height // block)), Image.Resampling.BILINEAR
        ).resize(image.size, Image.Resampling.NEAREST)
        return Image.composite(pixelated, image.convert("RGB"), mask.convert("L"))

    def generate(
        self,
        image: Image.Image,
        active_aus: Sequence[str] = (),
        trusted_face_crop: bool = False,
    ) -> List[GeneratedTwin]:
        active_aus = tuple(active_aus or ())
        if self.backend == "opencv":
            regions = self._opencv_regions(image, require_person_support=not trusted_face_crop)
        else:
            faces = self._detect(image)
            regions = [
                (
                    face_index,
                    group_name,
                    self._group_mask(image, face, components, group_name),
                    "MediaPipe FaceLandmarker contours",
                    1.0,
                    len(components),
                )
                for face_index, face in enumerate(faces)
                for group_name, components in LANDMARK_COMPONENTS.items()
            ]
        twins = []
        for face_index, group_name, mask, localiser, confidence, component_count in regions:
            target_aus = sorted(set(AU_GROUPS[group_name]) & set(active_aus))
            annotation_backed = bool(active_aus)
            is_control = annotation_backed and not target_aus
            twins.append(
                GeneratedTwin(
                    image=self._strong_ablation(image, mask),
                    mask=mask,
                    cue_family=self.cue_family,
                    operation=(
                        "strong_ablate_inactive_au_region_control"
                        if is_control
                        else "strong_ablate_annotated_au_region"
                        if annotation_backed
                        else "strong_ablate_au_related_region"
                    ),
                    target_region=mask_bbox(mask),
                    metadata={
                        "face_index": face_index,
                        "au_region": group_name,
                        "active_aus": list(active_aus),
                        "target_active_aus": target_aus,
                        "au_annotation_backed": annotation_backed,
                        "is_control": is_control,
                        "localiser": localiser,
                        "landmark_based": self.backend != "opencv",
                        "mask_component_count": component_count,
                        "ablation_method": "expanded_landmark_mask_strong_blur_and_pixelation",
                        "landmark_indices": [
                            index for component in LANDMARK_COMPONENTS[group_name] for index in component
                        ] if self.backend != "opencv" else [],
                        "face_confidence": confidence,
                        "trusted_face_crop": trusted_face_crop,
                    },
                )
            )
        return twins
