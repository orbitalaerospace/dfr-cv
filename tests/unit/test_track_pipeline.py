"""Integration tests for Detector -> ByteTracker -> Tracklet perception pipeline."""

from datetime import datetime, timedelta, timezone
from typing import List, Optional
import cv2
import numpy as np

from src.detectors.base import BaseDetector
from src.detectors.ultralytics_adapter import UltralyticsDetectorAdapter
from src.domain.enums import TargetClass, TrackState
from src.schemas.detection import BoundingBox, Detection
from src.schemas.tracking import Tracklet
from src.tracking.byte_tracker import ByteTracker


class MockSwimmerDetector(BaseDetector):
    """Deterministic mock detector simulating sequential aerial swimmer observations."""

    def __init__(self, detections_by_frame: List[List[Detection]]) -> None:
        self._detections = detections_by_frame
        self._classes = [TargetClass.SWIMMER]

    @property
    def detector_name(self) -> str:
        return "mock_swimmer_detector"

    @property
    def target_classes(self) -> List[TargetClass]:
        return self._classes

    def detect(
        self,
        image: np.ndarray,
        frame_id: int = 0,
        timestamp: Optional[datetime] = None,
    ) -> List[Detection]:
        if frame_id < len(self._detections):
            return self._detections[frame_id]
        return []


def test_detector_to_bytetrack_integration_pipeline():
    """Verify that any BaseDetector connects seamlessly to ByteTracker producing Tracklets."""
    t0 = datetime(2026, 9, 29, 12, 0, 0, tzinfo=timezone.utc)

    # 4 sequential frames of 2 moving swimmers
    frame_dets = [
        [
            Detection(
                bbox=BoundingBox(x1=100.0, y1=100.0, x2=130.0, y2=130.0),
                target_class=TargetClass.SWIMMER,
                confidence=0.88,
                frame_id=0,
                timestamp=t0,
            ),
            Detection(
                bbox=BoundingBox(x1=300.0, y1=200.0, x2=340.0, y2=240.0),
                target_class=TargetClass.SWIMMER,
                confidence=0.82,
                frame_id=0,
                timestamp=t0,
            ),
        ],
        [
            Detection(
                bbox=BoundingBox(x1=104.0, y1=102.0, x2=134.0, y2=132.0),
                target_class=TargetClass.SWIMMER,
                confidence=0.85,
                frame_id=1,
                timestamp=t0 + timedelta(milliseconds=100),
            ),
            Detection(
                bbox=BoundingBox(x1=305.0, y1=202.0, x2=345.0, y2=242.0),
                target_class=TargetClass.SWIMMER,
                confidence=0.80,
                frame_id=1,
                timestamp=t0 + timedelta(milliseconds=100),
            ),
        ],
        [
            # Swimmer 1 temporarily occluded (submerged under wave)
            Detection(
                bbox=BoundingBox(x1=310.0, y1=204.0, x2=350.0, y2=244.0),
                target_class=TargetClass.SWIMMER,
                confidence=0.83,
                frame_id=2,
                timestamp=t0 + timedelta(milliseconds=200),
            ),
        ],
        [
            # Swimmer 1 re-emerges
            Detection(
                bbox=BoundingBox(x1=112.0, y1=106.0, x2=142.0, y2=136.0),
                target_class=TargetClass.SWIMMER,
                confidence=0.86,
                frame_id=3,
                timestamp=t0 + timedelta(milliseconds=300),
            ),
            Detection(
                bbox=BoundingBox(x1=315.0, y1=206.0, x2=355.0, y2=246.0),
                target_class=TargetClass.SWIMMER,
                confidence=0.79,
                frame_id=3,
                timestamp=t0 + timedelta(milliseconds=300),
            ),
        ],
    ]

    detector = MockSwimmerDetector(frame_dets)
    tracker = ByteTracker(high_threshold=0.5, emit_coasting=True, min_hits=1)

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    all_frame_tracklets = []

    for f in range(4):
        # Frame processing pipeline: Detect -> Track -> Schema
        dets = detector.detect(dummy_frame, frame_id=f)
        tracklets = tracker.update(dets, frame_id=f)
        all_frame_tracklets.append(tracklets)

    # Frame 0: Two tracks initialized
    assert len(all_frame_tracklets[0]) == 2
    assert {t.track_id for t in all_frame_tracklets[0]} == {1, 2}

    # Frame 1: Identical IDs preserved
    assert len(all_frame_tracklets[1]) == 2
    assert {t.track_id for t in all_frame_tracklets[1]} == {1, 2}

    # Frame 2: Swimmer 1 coasting, Swimmer 2 tracked
    assert len(all_frame_tracklets[2]) == 2
    t_map = {t.track_id: t for t in all_frame_tracklets[2]}
    assert t_map[1].state == TrackState.COASTING
    assert t_map[2].state == TrackState.TRACKED

    # Frame 3: Swimmer 1 re-emerges, both tracked with same original IDs
    assert len(all_frame_tracklets[3]) == 2
    assert {t.track_id for t in all_frame_tracklets[3]} == {1, 2}
    assert all_frame_tracklets[3][0].state == TrackState.TRACKED
    assert all_frame_tracklets[3][1].state == TrackState.TRACKED


def test_real_detector_to_tracker_pipeline():
    """Verify live integration between UltralyticsDetectorAdapter and ByteTracker."""
    img = cv2.imread("data/samples/aerial_water_frame0.jpg")
    assert img is not None

    detector = UltralyticsDetectorAdapter(
        weights_path="models/seadronessee-yolov8n.pt",
        conf_threshold=0.25,
        device="cpu",
    )
    tracker = ByteTracker(high_threshold=0.35, min_hits=1)

    now = datetime(2026, 9, 29, 12, 0, 0, tzinfo=timezone.utc)
    dets = detector.detect(img, frame_id=0, timestamp=now)
    assert len(dets) > 0

    tracklets = tracker.update(dets, frame_id=0, timestamp=now)
    assert len(tracklets) > 0

    # Ensure all output objects are valid Tracklets
    for trk in tracklets:
        assert isinstance(trk, Tracklet)
        assert trk.track_id > 0
        assert trk.current_detection.target_class in (TargetClass.SWIMMER, TargetClass.WATERCRAFT)
