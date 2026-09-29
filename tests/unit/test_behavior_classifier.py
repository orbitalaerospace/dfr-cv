"""Unit tests for temporal distress behavior analysis and classification (Milestone 5)."""

from datetime import datetime, timedelta, timezone
import pytest

from src.analysis.behavior_classifier import TemporalDistressClassifier
from src.domain.enums import BehaviorState, TargetClass, TrackState
from src.schemas.behavior import BehaviorAssessment, BehaviorClassifierConfig
from src.schemas.detection import BoundingBox, Detection
from src.schemas.tracking import Tracklet


def make_detection(
    frame_id: int,
    center_x: float,
    center_y: float,
    width: float = 20.0,
    height: float = 20.0,
    confidence: float = 0.85,
    timestamp: datetime | None = None,
    target_class: TargetClass = TargetClass.SWIMMER,
    detector_name: str = "seadronessee_yolov8n",
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


class TestTemporalDistressClassifier:
    """Comprehensive unit test suite for TemporalDistressClassifier."""

    def test_insufficient_history_returns_unknown(self) -> None:
        """Test 1: Insufficient history (< min_observations) yields UNKNOWN state."""
        # Only 2 detections (min_observations is default 5)
        dets = [
            make_detection(frame_id=0, center_x=100.0, center_y=100.0),
            make_detection(frame_id=1, center_x=105.0, center_y=100.0),
        ]
        trk = make_tracklet(dets)

        classifier = TemporalDistressClassifier()
        assessment = classifier.update(trk)

        assert assessment.state == BehaviorState.UNKNOWN
        assert assessment.distress_score == 0.0
        assert assessment.observation_count == 2
        assert "Insufficient temporal observations" in assessment.explanation

    def test_stable_normal_motion_returns_normal(self) -> None:
        """Test 2: Steady, straight translational swimming yields NORMAL behavior state."""
        # Swimmer swims in a straight line: +5 px per frame at 10 FPS (50 px/s, high straightness)
        dets = [
            make_detection(frame_id=i, center_x=100.0 + i * 5.0, center_y=100.0, width=20.0, height=20.0)
            for i in range(15)
        ]
        trk = make_tracklet(dets)

        classifier = TemporalDistressClassifier()
        assessment = classifier.update(trk)

        assert assessment.state == BehaviorState.NORMAL
        assert assessment.distress_score < classifier.config.distress_threshold
        assert assessment.confidence >= 0.70
        assert "Normal swimming behavior" in assessment.explanation

    def test_multi_feature_abnormal_motion_triggers_candidate(self) -> None:
        """Test 3: Erratic motion, scale changes, and high direction changes trigger DISTRESS_CANDIDATE."""
        # Oscillating in place with rapid direction changes and area fluctuations:
        # Straightness ~ 0, direction change rate high, growth ratio high
        dets = []
        for i in range(12):
            dx = 15.0 if i % 2 == 0 else -15.0
            dy = 15.0 if (i // 2) % 2 == 0 else -15.0
            # Fluctuating scale: 15x15 to 40x40 (growth ratio > 2.5)
            w = 15.0 if i % 2 == 0 else 40.0
            h = 15.0 if i % 2 == 0 else 40.0
            dets.append(make_detection(frame_id=i, center_x=100.0 + dx, center_y=100.0 + dy, width=w, height=h))

        trk = make_tracklet(dets)
        classifier = TemporalDistressClassifier(config=BehaviorClassifierConfig(confirm_windows=3))
        assessment = classifier.update(trk)

        assert assessment.state == BehaviorState.DISTRESS_CANDIDATE
        assert assessment.distress_score >= classifier.config.distress_threshold
        assert assessment.consecutive_distress_windows == 1

    def test_persistent_abnormal_motion_confirms_distress(self) -> None:
        """Test 4: Sustained abnormal motion across consecutive windows escalates to DISTRESS_CONFIRMED."""
        classifier = TemporalDistressClassifier(config=BehaviorClassifierConfig(confirm_windows=3))

        dets = []
        t_base = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

        # Simulate 3 successive evaluation updates with erratic struggling in place
        for update_step in range(3):
            # Add 6 frames per step (0.5s duration each step)
            for i in range(6):
                f = update_step * 6 + i
                dx = 12.0 if f % 2 == 0 else -12.0
                dy = 12.0 if (f // 2) % 2 == 0 else -12.0
                w = 15.0 if f % 2 == 0 else 35.0
                h = 15.0 if f % 2 == 0 else 35.0
                ts = t_base + timedelta(seconds=f * 0.1)
                dets.append(make_detection(frame_id=f, center_x=200.0 + dx, center_y=200.0 + dy, width=w, height=h, timestamp=ts))

            trk = make_tracklet(dets)
            assessment = classifier.update(trk, current_timestamp=dets[-1].timestamp)

        assert assessment.state == BehaviorState.DISTRESS_CONFIRMED
        assert assessment.consecutive_distress_windows >= 3
        assert "Persistent distress indicators" in assessment.explanation

    def test_single_noisy_spike_does_not_trigger_distress(self) -> None:
        """Test 5: An isolated single-frame detector coordinate jump does not trigger distress."""
        # 10 frames of steady swimming, with a single noisy coordinate glitch at frame 6
        dets = []
        for i in range(12):
            if i == 6:
                # Isolated detector jump of 60 px
                cx, cy = 100.0 + i * 5.0, 160.0
            else:
                cx, cy = 100.0 + i * 5.0, 100.0
            dets.append(make_detection(frame_id=i, center_x=cx, center_y=cy, width=20.0, height=20.0))

        trk = make_tracklet(dets)
        classifier = TemporalDistressClassifier()
        assessment = classifier.update(trk)

        # Isolated spike should NOT reach the distress threshold (0.55)
        assert assessment.state == BehaviorState.NORMAL
        assert assessment.distress_score < classifier.config.distress_threshold

    def test_temporary_detection_gap_does_not_trigger_distress(self) -> None:
        """Test 6: A brief detection gap during calm swimming does not cause false distress."""
        # Frames 0..4 (calm), frame 5 missed, frames 6..10 (calm)
        dets = []
        for i in [0, 1, 2, 3, 4, 6, 7, 8, 9, 10]:
            dets.append(make_detection(frame_id=i, center_x=100.0 + i * 4.0, center_y=100.0))

        trk = make_tracklet(dets)
        classifier = TemporalDistressClassifier()
        assessment = classifier.update(trk)

        assert assessment.state == BehaviorState.NORMAL
        assert assessment.distress_score < classifier.config.distress_threshold

    def test_extreme_max_speed_outlier_does_not_dominate(self) -> None:
        """Test 7: Outlier peak speed does not dominate classification when straightness is high."""
        # Swimmer moving smoothly forward, but one frame has a brief burst (e.g. 500 px/s)
        # while moving in the forward direction with no erratic direction flips or area changes
        dets = [
            make_detection(frame_id=0, center_x=100.0, center_y=100.0),
            make_detection(frame_id=1, center_x=105.0, center_y=100.0),
            make_detection(frame_id=2, center_x=110.0, center_y=100.0),
            make_detection(frame_id=3, center_x=160.0, center_y=100.0),  # Sudden jump forward
            make_detection(frame_id=4, center_x=165.0, center_y=100.0),
            make_detection(frame_id=5, center_x=170.0, center_y=100.0),
            make_detection(frame_id=6, center_x=175.0, center_y=100.0),
        ]
        trk = make_tracklet(dets)
        classifier = TemporalDistressClassifier()
        assessment = classifier.update(trk)

        # High directional straightness prevents false distress
        assert assessment.state == BehaviorState.NORMAL
        assert assessment.distress_score < classifier.config.distress_threshold

    def test_state_transition_recovery_from_confirmed_distress(self) -> None:
        """Test 8: Confirmed distress requires recovery_windows consecutive normal windows to de-escalate."""
        classifier = TemporalDistressClassifier(config=BehaviorClassifierConfig(confirm_windows=2, recovery_windows=2, window_seconds=2.0))

        # First trigger confirmed distress with erratic motion
        erratic_dets = []
        for i in range(12):
            dx = 15.0 if i % 2 == 0 else -15.0
            dy = 15.0 if (i // 2) % 2 == 0 else -15.0
            w = 15.0 if i % 2 == 0 else 40.0
            erratic_dets.append(make_detection(frame_id=i, center_x=100.0 + dx, center_y=100.0 + dy, width=w, height=w))

        trk1 = make_tracklet(erratic_dets[:6])
        classifier.update(trk1)
        trk2 = make_tracklet(erratic_dets)
        assert classifier.update(trk2).state == BehaviorState.DISTRESS_CONFIRMED

        # Transition to normal straight swimming: append 35 calm frames (3.5s) to flush the 2.0s window
        t_now = erratic_dets[-1].timestamp
        calm_dets = list(erratic_dets)
        for i in range(1, 35):
            fid = 12 + i
            ts = t_now + timedelta(seconds=i * 0.1)
            calm_dets.append(make_detection(frame_id=fid, center_x=200.0 + i * 10.0, center_y=100.0, width=20.0, height=20.0, timestamp=ts))

        # Window at frame 37 (25 calm frames accumulated, purely calm in the 2.0s window)
        trk3 = make_tracklet(calm_dets[:38])
        assessment3 = classifier.update(trk3, current_timestamp=calm_dets[37].timestamp)
        assert assessment3.state == BehaviorState.DISTRESS_CONFIRMED
        assert "Maintaining distress confirmation" in assessment3.explanation

        # Next update with more calm frames -> meets recovery_windows requirement (2) -> de-escalates to NORMAL
        trk4 = make_tracklet(calm_dets)
        assessment4 = classifier.update(trk4, current_timestamp=calm_dets[-1].timestamp)
        assert assessment4.state == BehaviorState.NORMAL
        assert "Distress resolved after" in assessment4.explanation


    def test_multiple_independent_tracks(self) -> None:
        """Test 9: Multiple swimmer tracks are tracked and assessed independently without state cross-talk."""
        classifier = TemporalDistressClassifier()

        # Track 1: Normal straight swimming
        dets1 = [make_detection(frame_id=i, center_x=100.0 + i * 6.0, center_y=100.0) for i in range(10)]
        trk1 = make_tracklet(dets1, track_id=1)

        # Track 2: Erratic struggling in place
        dets2 = []
        for i in range(10):
            dx = 15.0 if i % 2 == 0 else -15.0
            dy = 15.0 if (i // 2) % 2 == 0 else -15.0
            w = 15.0 if i % 2 == 0 else 40.0
            dets2.append(make_detection(frame_id=i, center_x=300.0 + dx, center_y=300.0 + dy, width=w, height=w))
        trk2 = make_tracklet(dets2, track_id=2)

        assess1 = classifier.update(trk1)
        assess2 = classifier.update(trk2)

        assert assess1.track_id == 1
        assert assess1.state == BehaviorState.NORMAL
        assert assess2.track_id == 2
        assert assess2.state in (BehaviorState.DISTRESS_CANDIDATE, BehaviorState.DISTRESS_CONFIRMED)

    def test_non_swimmer_targets_are_exempt(self) -> None:
        """Test 10: Non-swimmer targets (e.g. WATERCRAFT, BUOY) are classified as NORMAL without distress calculation."""
        dets = [
            make_detection(frame_id=i, center_x=100.0, center_y=100.0, target_class=TargetClass.WATERCRAFT)
            for i in range(10)
        ]
        trk = make_tracklet(dets)
        classifier = TemporalDistressClassifier()
        assessment = classifier.update(trk)

        assert assessment.state == BehaviorState.NORMAL
        assert assessment.distress_score == 0.0
        assert "exempt from distress evaluation" in assessment.explanation

    def test_reset_functionality(self) -> None:
        """Test 11: Classifier reset clears history for specific track or all tracks."""
        classifier = TemporalDistressClassifier()
        dets = [make_detection(frame_id=i, center_x=100.0 + i * 5.0, center_y=100.0) for i in range(10)]
        trk = make_tracklet(dets, track_id=7)

        classifier.update(trk)
        assert classifier.get_assessment(7) is not None

        classifier.reset(track_id=7)
        assert classifier.get_assessment(7) is None

    def test_confidence_and_distress_score_computation(self) -> None:
        """Test 12: Behavioral confidence scales with observation density and distress score is bounded in [0, 1]."""
        classifier = TemporalDistressClassifier()
        # 20 detections in window -> high observation density
        dets = [make_detection(frame_id=i, center_x=100.0 + i * 5.0, center_y=100.0) for i in range(20)]
        trk = make_tracklet(dets)
        assessment = classifier.update(trk)

        assert 0.0 <= assessment.distress_score <= 1.0
        assert 0.0 <= assessment.confidence <= 1.0
        assert assessment.confidence >= 0.90  # high observation count yields high temporal confidence
        assert assessment.evidence is not None
        assert assessment.evidence.straightness_ratio > 0.8
        assert assessment.evidence.direction_score == 0.0

    def test_temporal_confirmation_window_count(self) -> None:
        """Test 13: Confirmation strictly enforces confirm_windows threshold."""
        confirm_needed = 4
        classifier = TemporalDistressClassifier(config=BehaviorClassifierConfig(confirm_windows=confirm_needed))

        # Generate erratic detections
        dets = []
        for i in range(25):
            dx = 15.0 if i % 2 == 0 else -15.0
            dy = 15.0 if (i // 2) % 2 == 0 else -15.0
            w = 15.0 if i % 2 == 0 else 40.0
            dets.append(make_detection(frame_id=i, center_x=100.0 + dx, center_y=100.0 + dy, width=w, height=w))

        # Windows 1 to confirm_needed - 1 must be DISTRESS_CANDIDATE
        for step in range(1, confirm_needed):
            sub_trk = make_tracklet(dets[: 6 + step * 4])
            assess = classifier.update(sub_trk)
            assert assess.state == BehaviorState.DISTRESS_CANDIDATE
            assert assess.consecutive_distress_windows == step

        # At step == confirm_needed, it must become DISTRESS_CONFIRMED
        sub_trk_final = make_tracklet(dets)
        assess_final = classifier.update(sub_trk_final)
        assert assess_final.state == BehaviorState.DISTRESS_CONFIRMED
        assert assess_final.consecutive_distress_windows >= confirm_needed

