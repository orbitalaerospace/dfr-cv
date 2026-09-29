"""Ultralytics YOLO detector adapter conforming to BaseDetector protocol."""

from datetime import datetime, timezone
import os
from pathlib import Path
import time
from typing import Dict, List, Optional
import numpy as np
import torch
from ultralytics import YOLO

from src.detectors.base import BaseDetector
from src.domain.enums import TargetClass
from src.schemas.detection import BoundingBox, Detection


# Default canonical mapping for standard 80-class COCO models
# Class 0: person -> TargetClass.PERSON (generic human, not water-specific)
# Class 8: boat   -> TargetClass.WATERCRAFT
DEFAULT_COCO_MAPPING: Dict[int, TargetClass] = {
    0: TargetClass.PERSON,
    8: TargetClass.WATERCRAFT,
}

# Default canonical mapping for 5-class SeaDronesSee models
# Native classes:
# 0: swimmer               -> TargetClass.SWIMMER
# 1: boat                  -> TargetClass.WATERCRAFT
# 2: jetski                -> TargetClass.WATERCRAFT
# 3: life_saving_appliances -> TargetClass.LIFE_SAVING_APPLIANCE
# 4: buoy                  -> TargetClass.BUOY
DEFAULT_SEADRONESSEE_MAPPING: Dict[int, TargetClass] = {
    0: TargetClass.SWIMMER,
    1: TargetClass.WATERCRAFT,
    2: TargetClass.WATERCRAFT,
    3: TargetClass.LIFE_SAVING_APPLIANCE,
    4: TargetClass.BUOY,
}


def resolve_model_class_mapping(model_names: Dict[int, str]) -> Dict[int, TargetClass]:
    """Auto-detect appropriate class mapping based on model native class names.

    Args:
        model_names: Dictionary of class IDs to class names from YOLO model.names.

    Returns:
        Canonical class mapping dictionary mapping integer class IDs to TargetClass enums.
    """
    names_lower = {int(idx): str(name).lower().strip() for idx, name in model_names.items()}

    # Check for SeaDronesSee signature
    if names_lower.get(0) == "swimmer" and names_lower.get(1) == "boat":
        return dict(DEFAULT_SEADRONESSEE_MAPPING)

    # Check for standard COCO signature
    if names_lower.get(0) == "person" and names_lower.get(8) == "boat":
        return dict(DEFAULT_COCO_MAPPING)

    # Name-based mapping fallback
    name_to_target = {
        "swimmer": TargetClass.SWIMMER,
        "boat": TargetClass.WATERCRAFT,
        "jetski": TargetClass.WATERCRAFT,
        "life_saving_appliances": TargetClass.LIFE_SAVING_APPLIANCE,
        "life_saving_appliance": TargetClass.LIFE_SAVING_APPLIANCE,
        "buoy": TargetClass.BUOY,
        "person": TargetClass.PERSON,
        "person_surface": TargetClass.PERSON_SURFACE,
        "floater": TargetClass.FLOATER,
        "life_jacket": TargetClass.LIFE_JACKET,
        "watercraft": TargetClass.WATERCRAFT,
    }
    mapping: Dict[int, TargetClass] = {}
    for idx, name in names_lower.items():
        if name in name_to_target:
            mapping[idx] = name_to_target[name]

    if mapping:
        return mapping

    return dict(DEFAULT_COCO_MAPPING)


def resolve_device(requested_device: str = "auto") -> str:
    """Resolve compute device string based on hardware availability."""
    req = requested_device.lower().strip()
    if req != "auto":
        return req

    if torch.cuda.is_available():
        return "cuda:0"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class UltralyticsDetectorAdapter(BaseDetector):
    """Adapter wrapping an Ultralytics YOLO model.

    Translates raw model tensor outputs into immutable M1 `Detection` schema objects,
    normalizes class taxonomies, and collects execution latency profiles.
    """

    def __init__(
        self,
        weights_path: str | Path,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str = "auto",
        class_mapping: Optional[Dict[int, TargetClass]] = None,
        filter_unmapped: bool = True,
        detector_name: str = "generic_detector_pipeline_validation",
    ) -> None:
        self.weights_path = Path(weights_path)
        if not self.weights_path.exists():
            raise FileNotFoundError(f"Detector weights file not found: {self.weights_path}")

        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = resolve_device(device)
        self.filter_unmapped = filter_unmapped
        self._detector_name = detector_name

        # Initialize model
        self.model = YOLO(str(self.weights_path))

        # Setup class taxonomy mapping
        if class_mapping is not None:
            self.class_mapping = dict(class_mapping)
        else:
            self.class_mapping = resolve_model_class_mapping(self.model.names)

        # Performance profiling store for the latest execution
        self.last_metrics: Dict[str, float] = {
            "preprocess_ms": 0.0,
            "inference_ms": 0.0,
            "postprocess_ms": 0.0,
            "total_ms": 0.0,
        }

    @property
    def detector_name(self) -> str:
        return self._detector_name

    @property
    def target_classes(self) -> List[TargetClass]:
        return sorted(list(set(self.class_mapping.values())), key=lambda c: c.value)

    def detect(
        self,
        image: np.ndarray,
        frame_id: int = 0,
        timestamp: Optional[datetime] = None,
    ) -> List[Detection]:
        if not isinstance(image, np.ndarray) or image.size == 0:
            raise ValueError("Input image must be a non-empty numpy array.")

        img_h, img_w = image.shape[:2]
        ts = timestamp if timestamp is not None else datetime.now(timezone.utc)

        t_start = time.perf_counter()

        # Run inference via Ultralytics engine
        results = self.model.predict(
            source=image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )

        t_end = time.perf_counter()
        self.last_metrics["total_ms"] = (t_end - t_start) * 1000.0

        if results and hasattr(results[0], "speed"):
            speed = results[0].speed
            self.last_metrics["preprocess_ms"] = speed.get("preprocess", 0.0)
            self.last_metrics["inference_ms"] = speed.get("inference", 0.0)
            self.last_metrics["postprocess_ms"] = speed.get("postprocess", 0.0)

        detections: List[Detection] = []
        if not results or len(results[0].boxes) == 0:
            return detections

        boxes = results[0].boxes
        xyxy_arr = boxes.xyxy.cpu().numpy()
        conf_arr = boxes.conf.cpu().numpy()
        cls_arr = boxes.cls.cpu().numpy().astype(int)

        for i in range(len(boxes)):
            raw_cls = int(cls_arr[i])
            conf = float(conf_arr[i])

            if raw_cls in self.class_mapping:
                target_cls = self.class_mapping[raw_cls]
            else:
                if self.filter_unmapped:
                    continue
                target_cls = TargetClass.UNKNOWN

            x1, y1, x2, y2 = xyxy_arr[i]

            # Clamp bounding box coordinates strictly within image boundaries
            x1 = max(0.0, min(float(x1), float(img_w - 1)))
            y1 = max(0.0, min(float(y1), float(img_h - 1)))
            x2 = max(0.0, min(float(x2), float(img_w)))
            y2 = max(0.0, min(float(y2), float(img_h)))

            # Discard degenerate zero/negative area boxes
            if x2 <= x1 or y2 <= y1:
                continue

            bbox = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)
            det = Detection(
                bbox=bbox,
                target_class=target_cls,
                confidence=conf,
                frame_id=frame_id,
                timestamp=ts,
                detector_name=self._detector_name,
            )
            detections.append(det)

        return detections
