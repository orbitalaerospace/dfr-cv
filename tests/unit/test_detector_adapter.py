"""Unit tests for detector adapters and pipeline interfaces."""

from datetime import datetime, timezone
import pytest
import numpy as np

from src.detectors.ultralytics_adapter import (
    DEFAULT_COCO_MAPPING,
    UltralyticsDetectorAdapter,
    resolve_device,
)
from src.domain.enums import TargetClass
from src.schemas.detection import Detection


def test_resolve_device():
    # Explicit devices pass through
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("cuda:0") == "cuda:0"
    assert resolve_device("mps") == "mps"

    # Auto resolution returns a valid device
    auto_dev = resolve_device("auto")
    assert auto_dev in ("cpu", "mps", "cuda:0")


def test_missing_weights_raises_file_not_found():
    with pytest.raises(FileNotFoundError, match="weights file not found"):
        UltralyticsDetectorAdapter("non_existent_weights.pt")


def test_coco_semantic_mapping_rules():
    """Ensure generic COCO classes are mapped honestly to generic domain classes."""
    # COCO class 0 ('person') MUST map to generic TargetClass.PERSON
    assert DEFAULT_COCO_MAPPING[0] == TargetClass.PERSON
    # COCO class 0 MUST NOT map to water-specific TargetClass.PERSON_SURFACE
    assert DEFAULT_COCO_MAPPING[0] != TargetClass.PERSON_SURFACE

    # COCO class 8 ('boat') maps to TargetClass.WATERCRAFT
    assert DEFAULT_COCO_MAPPING[8] == TargetClass.WATERCRAFT


def test_adapter_initialization():
    adapter = UltralyticsDetectorAdapter(
        weights_path="models/yolo11n.pt",
        conf_threshold=0.3,
        detector_name="test_detector",
    )
    assert adapter.detector_name == "test_detector"
    assert TargetClass.WATERCRAFT in adapter.target_classes
    assert TargetClass.PERSON in adapter.target_classes
    # Generic COCO adapter must not claim water-specific classes in its default targets
    assert TargetClass.PERSON_SURFACE not in adapter.target_classes
    assert TargetClass.SWIMMER not in adapter.target_classes


def test_unmapped_class_handling():
    """Verify adapter policy for classes not in the canonical mapping."""
    # filter_unmapped=True (default): unmapped classes are dropped
    adapter_filtered = UltralyticsDetectorAdapter(
        weights_path="models/yolo11n.pt",
        class_mapping={0: TargetClass.PERSON},
        filter_unmapped=True,
    )
    assert adapter_filtered.filter_unmapped is True

    # filter_unmapped=False: unmapped classes fall back to TargetClass.UNKNOWN
    adapter_unfiltered = UltralyticsDetectorAdapter(
        weights_path="models/yolo11n.pt",
        class_mapping={0: TargetClass.PERSON},
        filter_unmapped=False,
    )
    assert adapter_unfiltered.filter_unmapped is False


def test_detect_on_blank_frame_returns_empty_list():
    adapter = UltralyticsDetectorAdapter(
        weights_path="models/yolo11n.pt",
        conf_threshold=0.5,
        device="cpu",
    )
    # 300x300 pure black image
    blank = np.zeros((300, 300, 3), dtype=np.uint8)
    now = datetime.now(timezone.utc)
    dets = adapter.detect(blank, frame_id=10, timestamp=now)
    assert isinstance(dets, list)
    assert len(dets) == 0


def test_reject_empty_or_invalid_image():
    adapter = UltralyticsDetectorAdapter(
        weights_path="models/yolo11n.pt",
        device="cpu",
    )
    with pytest.raises(ValueError, match="non-empty numpy array"):
        adapter.detect(np.array([]))


def test_detect_on_real_sample_returns_valid_schema():
    import cv2
    img = cv2.imread("data/samples/aerial_water_frame0.jpg")
    assert img is not None, "Sample aerial image missing"

    adapter = UltralyticsDetectorAdapter(
        weights_path="models/yolo11n.pt",
        conf_threshold=0.20,
        device="cpu",
        detector_name="test_aerial_run",
    )
    now = datetime(2026, 9, 27, 12, 0, 0, tzinfo=timezone.utc)
    dets = adapter.detect(img, frame_id=5, timestamp=now)

    assert isinstance(dets, list)
    # Check that any detections produced are strictly valid M1 Detection instances
    for d in dets:
        assert isinstance(d, Detection)
        assert d.frame_id == 5
        assert d.timestamp == now
        assert d.detector_name == "test_aerial_run"
        assert 0.0 <= d.confidence <= 1.0
        assert d.bbox.x1 >= 0.0
        assert d.bbox.y1 >= 0.0
        assert d.bbox.x2 > d.bbox.x1
        assert d.bbox.y2 > d.bbox.y1
