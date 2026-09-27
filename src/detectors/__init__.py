"""Detector adapter package providing model-agnostic inference."""

from src.detectors.base import BaseDetector
from src.detectors.ultralytics_adapter import (
    DEFAULT_COCO_MAPPING,
    UltralyticsDetectorAdapter,
    resolve_device,
)

__all__ = [
    "BaseDetector",
    "DEFAULT_COCO_MAPPING",
    "UltralyticsDetectorAdapter",
    "resolve_device",
]
