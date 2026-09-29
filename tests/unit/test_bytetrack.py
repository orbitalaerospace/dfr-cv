"""Unit tests for ByteTrack multi-object tracking implementation."""

from datetime import datetime, timedelta, timezone
from typing import Optional
import pytest

from src.domain.enums import TargetClass, TrackState
from src.schemas.detection import BoundingBox, Detection
from src.tracking.byte_tracker import ByteTracker


def _make_det(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    frame_id: int,
    conf: float = 0.85,
    cls: TargetClass = TargetClass.SWIMMER,
    dt: Optional[datetime] = None,
) -> Detection:
    """Helper to synthesize deterministic Detection objects for testing."""
    base_time = datetime(2026, 9, 29, 12, 0, 0, tzinfo=timezone.utc)
    ts = dt if dt is not None else base_time + timedelta(milliseconds=100 * frame_id)
    return Detection(
        bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
        target_class=cls,
        confidence=conf,
        frame_id=frame_id,
        timestamp=ts,
        detector_name="unit_test_synthetic",
    )


def test_1_tracker_initialization():
    """1. Verify tracker initializes with expected defaults and empty state."""
    tracker = ByteTracker(
        high_threshold=0.5,
        low_threshold=0.1,
        match_threshold_high=0.8,
        match_threshold_low=0.5,
        max_lost=30,
        min_hits=3,
    )
    assert tracker.tracker_name == "bytetrack"
    assert tracker.high_threshold == 0.5
    assert tracker.low_threshold == 0.1
    assert tracker.max_lost == 30
    assert tracker.min_hits == 3


def test_2_first_detection_creates_track():
    """2. Verify that a first high-confidence detection creates a track."""
    tracker = ByteTracker(high_threshold=0.5, emit_unconfirmed=True)
    d0 = _make_det(100.0, 100.0, 140.0, 140.0, frame_id=0, conf=0.88)
    tracklets = tracker.update([d0], frame_id=0)

    assert len(tracklets) == 1
    t = tracklets[0]
    assert t.track_id == 1
    assert t.current_detection.bbox.as_xyxy() == (100.0, 100.0, 140.0, 140.0)
    assert t.current_detection.target_class == TargetClass.SWIMMER


def test_3_track_stable_id_across_consecutive_frames():
    """3. Verify that a stationary detection receives a stable ID across consecutive frames."""
    tracker = ByteTracker(high_threshold=0.5, min_hits=2)
    track_ids = []

    for f in range(5):
        det = _make_det(200.0, 200.0, 230.0, 230.0, frame_id=f, conf=0.85)
        tracklets = tracker.update([det], frame_id=f)
        assert len(tracklets) == 1
        track_ids.append(tracklets[0].track_id)

    # Must preserve identity 1 across all 5 frames
    assert track_ids == [1, 1, 1, 1, 1]


def test_4_two_detections_create_two_distinct_tracks():
    """4. Verify that two spatially separated detections create two distinct track IDs."""
    tracker = ByteTracker(high_threshold=0.5)
    d_a = _make_det(50.0, 50.0, 80.0, 80.0, frame_id=0)
    d_b = _make_det(300.0, 300.0, 340.0, 340.0, frame_id=0)

    tracklets = tracker.update([d_a, d_b], frame_id=0)
    assert len(tracklets) == 2
    ids = {t.track_id for t in tracklets}
    assert ids == {1, 2}


def test_5_tracks_remain_associated_when_objects_move_smoothly():
    """5. Verify that targets moving with smooth linear translation maintain stable IDs."""
    tracker = ByteTracker(high_threshold=0.5)

    for f in range(10):
        # Swimmer 1 moves diagonally at (5, 3) px/frame
        x1_a, y1_a = 50.0 + f * 5.0, 50.0 + f * 3.0
        d_a = _make_det(x1_a, y1_a, x1_a + 30.0, y1_a + 30.0, frame_id=f)

        # Swimmer 2 moves horizontally at (-4, 0) px/frame
        x1_b, y1_b = 400.0 - f * 4.0, 200.0
        d_b = _make_det(x1_b, y1_b, x1_b + 35.0, y1_b + 35.0, frame_id=f)

        tracklets = tracker.update([d_a, d_b], frame_id=f)
        assert len(tracklets) == 2
        id_map = {t.track_id: t for t in tracklets}
        assert 1 in id_map
        assert 2 in id_map


def test_6_high_confidence_association():
    """6. Verify Stage 1 primary association on high-confidence detections."""
    tracker = ByteTracker(high_threshold=0.6)
    d0 = _make_det(100.0, 100.0, 130.0, 130.0, frame_id=0, conf=0.90)
    d1 = _make_det(102.0, 101.0, 132.0, 131.0, frame_id=1, conf=0.85)

    tracker.update([d0], frame_id=0)
    t1 = tracker.update([d1], frame_id=1)

    assert len(t1) == 1
    assert t1[0].track_id == 1
    assert t1[0].current_detection.confidence == 0.85


def test_7_low_confidence_second_stage_association():
    """7. Verify Stage 2 recovery association on low-confidence detections (occlusion/submersion)."""
    tracker = ByteTracker(high_threshold=0.6, low_threshold=0.2, min_hits=1)

    # Frame 0: High-confidence detection initializes confirmed track
    d0 = _make_det(100.0, 100.0, 130.0, 130.0, frame_id=0, conf=0.85)
    t0 = tracker.update([d0], frame_id=0)
    assert t0[0].track_id == 1

    # Frame 1: Target partially submerged, confidence drops to 0.35 (< high_thresh 0.6)
    d1_low = _make_det(103.0, 102.0, 133.0, 132.0, frame_id=1, conf=0.35)
    t1 = tracker.update([d1_low], frame_id=1)

    # ByteTrack must successfully associate the low-confidence detection in Stage 2
    assert len(t1) == 1
    assert t1[0].track_id == 1
    assert t1[0].current_detection.confidence == 0.35
    assert len(t1[0].observation_history) == 2


def test_8_temporary_detection_dropout_produces_coasting():
    """8. Verify that a missed detection in an active track transitions to COASTING state."""
    tracker = ByteTracker(high_threshold=0.5, emit_coasting=True, min_hits=1)

    # Frame 0: Tracked
    d0 = _make_det(100.0, 100.0, 130.0, 130.0, frame_id=0, conf=0.80)
    t0 = tracker.update([d0], frame_id=0)
    assert t0[0].state == TrackState.TRACKED

    # Frame 1: Swimmer submerged under wave crest (empty detection list)
    t1 = tracker.update([], frame_id=1)

    assert len(t1) == 1
    assert t1[0].track_id == 1
    assert t1[0].state == TrackState.COASTING
    # Coasting tracklet must emit a valid non-empty predicted bounding box
    assert t1[0].current_detection.bbox.width > 0
    assert t1[0].current_detection.bbox.height > 0
    assert t1[0].current_detection.detector_name == "kalman_coasting"


def test_9_detection_returning_after_dropout_reconnects_to_same_id():
    """9. Verify that a target reappearing after multi-frame dropout reclaims its original track ID."""
    tracker = ByteTracker(high_threshold=0.5, max_lost=10, min_hits=1)

    # Frame 0: Detected & Tracked
    d0 = _make_det(100.0, 100.0, 140.0, 140.0, frame_id=0, conf=0.85)
    tracker.update([d0], frame_id=0)

    # Frames 1-3: Dropped (coasting through wave chop)
    tracker.update([], frame_id=1)
    tracker.update([], frame_id=2)
    tracker.update([], frame_id=3)

    # Frame 4: Target re-emerges at water surface near projected position
    d4 = _make_det(108.0, 106.0, 148.0, 146.0, frame_id=4, conf=0.80)
    t4 = tracker.update([d4], frame_id=4)

    assert len(t4) == 1
    # Must preserve identity 1 and return to TRACKED state
    assert t4[0].track_id == 1
    assert t4[0].state == TrackState.TRACKED
    assert t4[0].current_detection.confidence == 0.80


def test_10_track_pruned_after_max_lost():
    """10. Verify that a lost track is pruned from active output after max_lost frames."""
    tracker = ByteTracker(high_threshold=0.5, max_lost=3, emit_coasting=True, min_hits=1)

    # Frame 0: Active
    d0 = _make_det(100.0, 100.0, 130.0, 130.0, frame_id=0, conf=0.80)
    tracker.update([d0], frame_id=0)

    # Frames 1, 2, 3: Coasting (3 missing frames == max_lost)
    assert len(tracker.update([], frame_id=1)) == 1
    assert len(tracker.update([], frame_id=2)) == 1
    assert len(tracker.update([], frame_id=3)) == 1

    # Frame 4: Exceeds max_lost (4th missing frame) -> track marked LOST and pruned
    t4 = tracker.update([], frame_id=4)
    assert len(t4) == 0


def test_11_observation_history_grows_correctly():
    """11. Verify that observation_history accurately accumulates chronological detections."""
    tracker = ByteTracker(high_threshold=0.5, min_hits=1)

    for f in range(6):
        d = _make_det(100.0 + f * 2.0, 100.0, 140.0 + f * 2.0, 140.0, frame_id=f)
        tracklets = tracker.update([d], frame_id=f)

    assert len(tracklets) == 1
    t = tracklets[0]
    assert len(t.observation_history) == 6
    assert t.observation_count == 6
    # Verify chronological frame IDs in history
    hist_frames = [obs.frame_id for obs in t.observation_history]
    assert hist_frames == [0, 1, 2, 3, 4, 5]


def test_12_tracklet_invariants_remain_valid():
    """12. Verify all Pydantic Tracklet domain contract invariants hold."""
    tracker = ByteTracker(high_threshold=0.5, min_hits=1)

    t0 = datetime(2026, 9, 29, 12, 0, 0, tzinfo=timezone.utc)
    t1 = datetime(2026, 9, 29, 12, 0, 2, tzinfo=timezone.utc)

    d0 = _make_det(50.0, 50.0, 90.0, 90.0, frame_id=0, dt=t0)
    d1 = _make_det(55.0, 52.0, 95.0, 92.0, frame_id=20, dt=t1)

    tracker.update([d0], frame_id=0, timestamp=t0)
    tracklets = tracker.update([d1], frame_id=20, timestamp=t1)

    t = tracklets[0]
    assert t.track_id > 0
    assert t.last_seen_timestamp >= t.first_seen_timestamp
    assert t.lifespan_seconds == pytest.approx(2.0)
    assert t.observation_count == 2
    assert t.current_detection.bbox.width > 0
    assert t.current_detection.bbox.height > 0


def test_13_enforce_class_matching_prevents_swaps():
    """13. Verify that swimmer and boat in overlapping coordinates do not swap identities."""
    tracker = ByteTracker(high_threshold=0.5, enforce_class_match=True, min_hits=1)

    # Frame 0: Swimmer at (100, 100, 140, 140) -> Track ID 1
    d_swimmer = _make_det(100.0, 100.0, 140.0, 140.0, frame_id=0, cls=TargetClass.SWIMMER)
    t0 = tracker.update([d_swimmer], frame_id=0)
    assert t0[0].track_id == 1
    assert t0[0].current_detection.target_class == TargetClass.SWIMMER

    # Frame 1: Boat arrives in overlapping position (102, 102, 142, 142)
    d_boat = _make_det(102.0, 102.0, 142.0, 142.0, frame_id=1, cls=TargetClass.WATERCRAFT)
    t1 = tracker.update([d_boat], frame_id=1)

    # The boat must NOT steal Swimmer ID 1; it must initialize a new distinct track ID 2
    classes_by_id = {trk.track_id: trk.current_detection.target_class for trk in t1}
    assert 2 in classes_by_id
    assert classes_by_id[2] == TargetClass.WATERCRAFT
