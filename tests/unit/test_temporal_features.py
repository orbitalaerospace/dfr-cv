"""Unit and integration tests for temporal feature extraction (Milestone 4.5)."""

from datetime import datetime, timedelta, timezone
import pytest

from src.analysis.temporal_features import (
    extract_temporal_features,
    format_track_diagnostics,
)
from src.domain.enums import TargetClass, TrackState
from src.schemas.detection import BoundingBox, Detection
from src.schemas.temporal_features import TemporalFeatureConfig, TemporalFeatures
from src.schemas.tracking import Tracklet
from src.tracking.byte_tracker import ByteTracker


def make_detection(
    frame_id: int,
    center_x: float,
    center_y: float,
    width: float = 20.0,
    height: float = 20.0,
    confidence: float = 0.8,
    timestamp: datetime | None = None,
    target_class: TargetClass = TargetClass.SWIMMER,
    detector_name: str = "synthetic_detector",
) -> Detection:
    """Helper to synthesize canonical Detection instances."""
    x1 = max(0.0, center_x - width / 2.0)
    y1 = max(0.0, center_y - height / 2.0)
    x2 = x1 + width
    y2 = y1 + height
    ts = timestamp or datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc) + timedelta(seconds=frame_id * 0.1)
    return Detection(
        bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
        target_class=target_class,
        confidence=confidence,
        frame_id=frame_id,
        timestamp=ts,
        detector_name=detector_name,
    )


def make_tracklet(
    detections: list[Detection],
    track_id: int = 1,
    state: TrackState = TrackState.TRACKED,
) -> Tracklet:
    """Helper to synthesize domain Tracklet instances from a sequence of detections."""
    if not detections:
        raise ValueError("Cannot make tracklet with zero detections.")
    return Tracklet(
        track_id=track_id,
        state=state,
        current_detection=detections[-1],
        first_seen_timestamp=detections[0].timestamp,
        last_seen_timestamp=detections[-1].timestamp,
        observation_history=tuple(detections),
    )


class TestTemporalFeatureExtraction:
    """Unit test suite for pure temporal feature extraction."""

    def test_single_stationary_track(self) -> None:
        """Test 1: Single stationary tracklet produces zero displacement, speed, and acceleration."""
        # 10 frames at center (50, 50)
        dets = [make_detection(frame_id=i, center_x=50.0, center_y=50.0) for i in range(10)]
        trk = make_tracklet(dets)

        features = extract_temporal_features(trk)

        assert features.track_id == 1
        assert features.observation_count == 10
        assert features.total_frames == 10
        assert features.visibility_ratio == 1.0
        assert features.detection_gap_count == 0
        assert features.max_consecutive_gap == 0
        assert features.start_centroid == (50.0, 50.0)
        assert features.end_centroid == (50.0, 50.0)
        assert features.total_displacement_px == 0.0
        assert features.total_path_length_px == 0.0
        assert features.mean_speed_px_per_sec == 0.0
        assert features.max_speed_px_per_sec == 0.0
        assert features.mean_acceleration_px_per_sec2 == 0.0
        assert features.direction_change_count == 0

    def test_constant_speed_movement(self) -> None:
        """Test 2: Movement at constant speed produces analytically exact speed and zero acceleration."""
        # Moves (20, 20), (30, 20), (40, 20) at 10 FPS (dt = 0.1s, dx = 10px -> 100 px/s)
        dets = [
            make_detection(frame_id=0, center_x=20.0, center_y=20.0),
            make_detection(frame_id=1, center_x=30.0, center_y=20.0),
            make_detection(frame_id=2, center_x=40.0, center_y=20.0),
        ]
        trk = make_tracklet(dets)

        features = extract_temporal_features(trk)

        assert features.start_centroid == (20.0, 20.0)
        assert features.end_centroid == (40.0, 20.0)
        assert features.total_displacement_px == pytest.approx(20.0, abs=0.05)
        assert features.total_path_length_px == pytest.approx(20.0, abs=0.05)
        assert features.mean_speed_px_per_sec == pytest.approx(100.0, abs=0.5)
        assert features.max_speed_px_per_sec == pytest.approx(100.0, abs=0.5)
        assert features.mean_acceleration_px_per_sec2 == pytest.approx(0.0, abs=0.05)
        assert features.direction_change_count == 0

    def test_increasing_bbox_size(self) -> None:
        """Test 3: Increasing bounding box area correctly computes min, max, and growth ratio."""
        dets = [
            make_detection(frame_id=0, center_x=50.0, center_y=50.0, width=10.0, height=10.0),  # area = 100
            make_detection(frame_id=1, center_x=50.0, center_y=50.0, width=20.0, height=20.0),  # area = 400
            make_detection(frame_id=2, center_x=50.0, center_y=50.0, width=30.0, height=30.0),  # area = 900
        ]
        trk = make_tracklet(dets)

        features = extract_temporal_features(trk)

        assert features.min_bbox_area_px2 == pytest.approx(100.0, abs=0.1)
        assert features.max_bbox_area_px2 == pytest.approx(900.0, abs=0.1)
        assert features.bbox_growth_ratio == pytest.approx(9.0, abs=0.05)

    def test_temporary_detection_gap(self) -> None:
        """Test 4: Single temporary detection gap correctly records gap count and max gap."""
        # Detections at frames 0, 1, 4 (missing 2 and 3 -> 1 gap event of 2 frames)
        dets = [
            make_detection(frame_id=0, center_x=10.0, center_y=10.0),
            make_detection(frame_id=1, center_x=20.0, center_y=10.0),
            make_detection(frame_id=4, center_x=50.0, center_y=10.0),
        ]
        trk = make_tracklet(dets)

        features = extract_temporal_features(trk)

        assert features.observation_count == 3
        assert features.total_frames == 5  # frames 0, 1, 2, 3, 4
        assert features.visibility_ratio == pytest.approx(3 / 5, abs=0.01)
        assert features.detection_gap_count == 1
        assert features.max_consecutive_gap == 2

    def test_multiple_gaps(self) -> None:
        """Test 5: Multiple distinct gaps are tracked with accurate total and max length."""
        # Detections at frames 0, 1, 4, 7, 8
        # Gap 1: frames 2, 3 (length 2)
        # Gap 2: frames 5, 6 (length 2)
        dets = [
            make_detection(frame_id=0, center_x=10.0, center_y=10.0),
            make_detection(frame_id=1, center_x=20.0, center_y=10.0),
            make_detection(frame_id=4, center_x=30.0, center_y=10.0),
            make_detection(frame_id=7, center_x=40.0, center_y=10.0),
            make_detection(frame_id=8, center_x=50.0, center_y=10.0),
        ]
        trk = make_tracklet(dets)

        features = extract_temporal_features(trk)

        assert features.observation_count == 5
        assert features.total_frames == 9  # frames 0..8
        assert features.detection_gap_count == 2
        assert features.max_consecutive_gap == 2
        assert features.visibility_ratio == pytest.approx(5 / 9, abs=0.01)

    def test_direction_changes(self) -> None:
        """Test 6: Distinct 90-degree and 180-degree turns count as significant direction changes."""
        # Step 1: (10, 10) -> (30, 10) [dx=+20, dy=0]
        # Step 2: (30, 10) -> (30, 30) [dx=0, dy=+20] -> 90 deg turn
        # Step 3: (30, 30) -> (50, 30) [dx=+20, dy=0] -> 90 deg turn
        dets = [
            make_detection(frame_id=0, center_x=10.0, center_y=10.0),
            make_detection(frame_id=1, center_x=30.0, center_y=10.0),
            make_detection(frame_id=2, center_x=30.0, center_y=30.0),
            make_detection(frame_id=3, center_x=50.0, center_y=30.0),
        ]
        trk = make_tracklet(dets)

        features = extract_temporal_features(trk)

        assert features.direction_change_count == 2
        assert features.total_displacement_px == pytest.approx(
            ((50 - 10) ** 2 + (30 - 10) ** 2) ** 0.5, abs=0.05
        )
        assert features.total_path_length_px == pytest.approx(60.0, abs=0.05)

    def test_zero_near_zero_movement(self) -> None:
        """Test 7: Small detector jiggles below min_movement_px do not trigger false direction changes."""
        # Jiggles within subpixel / small 0.5px movements
        dets = [
            make_detection(frame_id=0, center_x=100.0, center_y=100.0),
            make_detection(frame_id=1, center_x=100.5, center_y=100.2),
            make_detection(frame_id=2, center_x=100.1, center_y=100.6),
            make_detection(frame_id=3, center_x=100.3, center_y=100.1),
        ]
        trk = make_tracklet(dets)

        config = TemporalFeatureConfig(min_movement_px=2.0)
        features = extract_temporal_features(trk, config=config)

        assert features.direction_change_count == 0

    def test_single_observation(self) -> None:
        """Test 8: Tracklet with exactly one observation produces neutral zero-motion metrics."""
        det = make_detection(frame_id=0, center_x=150.0, center_y=200.0, width=20.0, height=30.0, confidence=0.85)
        trk = make_tracklet([det])

        features = extract_temporal_features(trk)

        assert features.observation_count == 1
        assert features.total_frames == 1
        assert features.duration_seconds == 0.0
        assert features.visibility_ratio == 1.0
        assert features.detection_gap_count == 0
        assert features.max_consecutive_gap == 0
        assert features.start_centroid == (150.0, 200.0)
        assert features.end_centroid == (150.0, 200.0)
        assert features.total_displacement_px == 0.0
        assert features.mean_speed_px_per_sec == 0.0
        assert features.max_speed_px_per_sec == 0.0
        assert features.mean_acceleration_px_per_sec2 == 0.0
        assert features.direction_change_count == 0
        assert features.min_bbox_area_px2 == pytest.approx(600.0, abs=0.1)
        assert features.max_bbox_area_px2 == pytest.approx(600.0, abs=0.1)
        assert features.bbox_growth_ratio == 1.0
        assert features.mean_confidence == pytest.approx(0.85, abs=0.01)

    def test_empty_observation_history_fallback(self) -> None:
        """Test 9: Tracklet where observation_history is empty evaluates current_detection correctly."""
        det = make_detection(frame_id=5, center_x=80.0, center_y=90.0)
        trk = Tracklet(
            track_id=42,
            state=TrackState.NEW,
            current_detection=det,
            first_seen_timestamp=det.timestamp,
            last_seen_timestamp=det.timestamp,
            observation_history=(),
        )

        features = extract_temporal_features(trk)

        assert features.track_id == 42
        assert features.observation_count == 1
        assert features.total_frames == 1
        assert features.start_centroid == (80.0, 90.0)
        assert features.end_centroid == (80.0, 90.0)

    def test_coasting_observation_handling(self) -> None:
        """Test 10: Coasting observations are excluded from real counts and confidence aggregation."""
        real_dets = [
            make_detection(frame_id=0, center_x=10.0, center_y=10.0, confidence=0.8, detector_name="detector"),
            make_detection(frame_id=1, center_x=20.0, center_y=10.0, confidence=0.9, detector_name="detector"),
        ]
        coasting_det = make_detection(
            frame_id=3,
            center_x=40.0,
            center_y=10.0,
            confidence=0.25,
            detector_name="kalman_coasting",
        )

        trk = Tracklet(
            track_id=7,
            state=TrackState.COASTING,
            current_detection=coasting_det,
            first_seen_timestamp=real_dets[0].timestamp,
            last_seen_timestamp=real_dets[-1].timestamp,
            observation_history=(real_dets[0], real_dets[1], coasting_det),
        )

        features = extract_temporal_features(trk)

        # Coasting prediction must NOT be counted as a real observation
        assert features.observation_count == 2
        # Total frame span is 0 to 3 -> 4 frames
        assert features.total_frames == 4
        assert features.visibility_ratio == pytest.approx(2 / 4, abs=0.01)
        # Gap between frame 1 and frame 3 is 2 frames
        assert features.detection_gap_count == 1
        assert features.max_consecutive_gap == 2
        # Confidence must ONLY average real detections: (0.8 + 0.9) / 2 = 0.85
        assert features.mean_confidence == pytest.approx(0.85, abs=0.01)

    def test_confidence_aggregation(self) -> None:
        """Test 11: Real detection confidences are aggregated accurately."""
        dets = [
            make_detection(frame_id=0, center_x=10.0, center_y=10.0, confidence=0.60),
            make_detection(frame_id=1, center_x=20.0, center_y=10.0, confidence=0.80),
            make_detection(frame_id=2, center_x=30.0, center_y=10.0, confidence=1.00),
        ]
        trk = make_tracklet(dets)

        features = extract_temporal_features(trk)

        assert features.mean_confidence == pytest.approx(0.80, abs=0.001)

    def test_timestamp_and_fps_fallback(self) -> None:
        """Test 12: Speed is computed via explicit timestamps, or via FPS when timestamps are identical."""
        # 12a: Explicit timestamps
        t0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        t1 = t0 + timedelta(seconds=0.5)  # dt = 0.5s
        dets_time = [
            make_detection(frame_id=0, center_x=100.0, center_y=100.0, timestamp=t0),
            make_detection(frame_id=1, center_x=150.0, center_y=100.0, timestamp=t1),
        ]
        trk_time = make_tracklet(dets_time)
        feat_time = extract_temporal_features(trk_time)
        # 50 px / 0.5 s = 100 px/s
        assert feat_time.mean_speed_px_per_sec == pytest.approx(100.0, abs=0.1)

        # 12b: Identical timestamps (dt = 0) fallback to FPS
        dets_fps = [
            make_detection(frame_id=0, center_x=100.0, center_y=100.0, timestamp=t0),
            make_detection(frame_id=2, center_x=120.0, center_y=100.0, timestamp=t0),  # frame_diff = 2
        ]
        trk_fps = make_tracklet(dets_fps)
        config = TemporalFeatureConfig(default_fps=10.0)
        feat_fps = extract_temporal_features(trk_fps, config=config)
        # frame_diff = 2 at 10 FPS -> dt = 0.2 s, distance = 20 px -> 100 px/s
        assert feat_fps.mean_speed_px_per_sec == pytest.approx(100.0, abs=0.1)


    def test_format_track_diagnostics(self) -> None:
        """Test diagnostic string formatter produces clean human-readable output."""
        dets = [
            make_detection(frame_id=0, center_x=100.0, center_y=100.0, confidence=0.8),
            make_detection(frame_id=1, center_x=120.0, center_y=100.0, confidence=0.9),
        ]
        trk = make_tracklet(dets)
        features = extract_temporal_features(trk)
        text = format_track_diagnostics(features)

        assert "Track 1" in text
        assert "SWIMMER" in text
        assert "Observations:" in text
        assert "Total displacement:" in text
        assert "px/s" in text


class TestByteTrackToTemporalFeaturesIntegration:
    """Integration test: Synthetic Detections -> ByteTracker -> Tracklet -> TemporalFeatures."""

    def test_full_pipeline_association_to_feature_extraction(self) -> None:
        """Verify that real M4 ByteTracker output directly feeds the M4.5 feature extraction layer."""
        tracker = ByteTracker(
            high_threshold=0.4,
            low_threshold=0.1,
            match_threshold_high=0.8,
            min_hits=2,
            max_lost=5,
        )

        t_base = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        # Feed 5 sequential frames of a moving target
        # Target starts at (50, 50) and moves +10px in x each frame at 10 FPS (100 px/s)
        confirmed_tracklets = []
        for f in range(5):
            ts = t_base + timedelta(seconds=f * 0.1)
            det = make_detection(
                frame_id=f,
                center_x=50.0 + f * 10.0,
                center_y=50.0,
                width=30.0,
                height=30.0,
                confidence=0.85,
                timestamp=ts,
            )
            tracklets = tracker.update([det], frame_id=f, timestamp=ts)
            for trk in tracklets:
                if trk.state == TrackState.TRACKED:
                    confirmed_tracklets.append(trk)

        assert len(confirmed_tracklets) > 0, "ByteTracker should have confirmed the track"
        final_tracklet = confirmed_tracklets[-1]

        # Feed directly into temporal feature extractor
        features = extract_temporal_features(final_tracklet)

        assert isinstance(features, TemporalFeatures)
        assert features.track_id == final_tracklet.track_id
        assert features.target_class == TargetClass.SWIMMER
        assert features.observation_count >= 4
        assert features.visibility_ratio == 1.0
        assert features.detection_gap_count == 0
        assert features.start_centroid == pytest.approx((50.0, 50.0), abs=1.0)
        assert features.mean_speed_px_per_sec == pytest.approx(100.0, abs=5.0)
        assert features.mean_confidence == pytest.approx(0.85, abs=0.05)
        assert features.direction_change_count == 0
