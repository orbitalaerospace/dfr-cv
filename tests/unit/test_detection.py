"""Unit tests for spatial detection data contracts."""

from datetime import datetime, timezone
import pytest
from pydantic import ValidationError

from src.domain.enums import TargetClass
from src.schemas.detection import BoundingBox, Detection


def test_valid_bounding_box():
    bbox = BoundingBox(x1=10.0, y1=20.0, x2=110.0, y2=120.0)
    assert bbox.x1 == 10.0
    assert bbox.y1 == 20.0
    assert bbox.x2 == 110.0
    assert bbox.y2 == 120.0
    assert bbox.width == 100.0
    assert bbox.height == 100.0
    assert bbox.area == 10000.0
    assert bbox.center == (60.0, 70.0)
    assert bbox.aspect_ratio == 1.0
    assert bbox.as_xyxy() == (10.0, 20.0, 110.0, 120.0)
    assert bbox.as_xywh() == (10.0, 20.0, 100.0, 100.0)


def test_reject_negative_bbox_coordinates():
    with pytest.raises(ValidationError, match="non-negative"):
        BoundingBox(x1=-1.0, y1=10.0, x2=50.0, y2=50.0)

    with pytest.raises(ValidationError, match="non-negative"):
        BoundingBox(x1=10.0, y1=-5.0, x2=50.0, y2=50.0)


def test_reject_zero_or_negative_bbox_dimensions():
    # x2 equal to x1 (zero width)
    with pytest.raises(ValidationError, match="Invalid box width"):
        BoundingBox(x1=20.0, y1=10.0, x2=20.0, y2=50.0)

    # x2 less than x1 (negative width)
    with pytest.raises(ValidationError, match="Invalid box width"):
        BoundingBox(x1=50.0, y1=10.0, x2=20.0, y2=50.0)

    # y2 equal to y1 (zero height)
    with pytest.raises(ValidationError, match="Invalid box height"):
        BoundingBox(x1=10.0, y1=30.0, x2=50.0, y2=30.0)

    # y2 less than y1 (negative height)
    with pytest.raises(ValidationError, match="Invalid box height"):
        BoundingBox(x1=10.0, y1=50.0, x2=50.0, y2=30.0)


def test_valid_detection():
    now = datetime.now(timezone.utc)
    bbox = BoundingBox(x1=50.0, y1=60.0, x2=150.0, y2=160.0)
    det = Detection(
        bbox=bbox,
        target_class=TargetClass.SWIMMER,
        confidence=0.88,
        frame_id=42,
        timestamp=now,
        detector_name="test_model",
    )
    assert det.bbox == bbox
    assert det.target_class == TargetClass.SWIMMER
    assert det.confidence == 0.88
    assert det.frame_id == 42
    assert det.timestamp == now
    assert det.detector_name == "test_model"


def test_reject_confidence_out_of_bounds():
    now = datetime.now(timezone.utc)
    bbox = BoundingBox(x1=10.0, y1=10.0, x2=20.0, y2=20.0)

    with pytest.raises(ValidationError):
        Detection(
            bbox=bbox,
            target_class=TargetClass.SWIMMER,
            confidence=-0.01,
            frame_id=1,
            timestamp=now,
        )

    with pytest.raises(ValidationError):
        Detection(
            bbox=bbox,
            target_class=TargetClass.SWIMMER,
            confidence=1.01,
            frame_id=1,
            timestamp=now,
        )


def test_reject_negative_frame_id():
    now = datetime.now(timezone.utc)
    bbox = BoundingBox(x1=10.0, y1=10.0, x2=20.0, y2=20.0)

    with pytest.raises(ValidationError):
        Detection(
            bbox=bbox,
            target_class=TargetClass.PERSON_SURFACE,
            confidence=0.9,
            frame_id=-1,
            timestamp=now,
        )


def test_detection_immutability():
    now = datetime.now(timezone.utc)
    bbox = BoundingBox(x1=10.0, y1=10.0, x2=20.0, y2=20.0)
    det = Detection(
        bbox=bbox,
        target_class=TargetClass.FLOATER,
        confidence=0.75,
        frame_id=5,
        timestamp=now,
    )
    with pytest.raises(ValidationError):
        det.confidence = 0.95  # Frozen model rejects mutation
