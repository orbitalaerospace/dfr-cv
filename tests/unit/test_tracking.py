"""Unit tests for tracking and tracklet data contracts."""

from datetime import datetime, timedelta, timezone
import pytest
from pydantic import ValidationError

from src.domain.enums import TargetClass, TrackState
from src.schemas.detection import BoundingBox, Detection
from src.schemas.tracking import Tracklet


def _create_detection(frame_id: int, dt: datetime, conf: float = 0.8) -> Detection:
    return Detection(
        bbox=BoundingBox(x1=10.0, y1=10.0, x2=50.0, y2=50.0),
        target_class=TargetClass.SWIMMER,
        confidence=conf,
        frame_id=frame_id,
        timestamp=dt,
    )


def test_valid_tracklet():
    t0 = datetime(2026, 9, 27, 12, 0, 0, tzinfo=timezone.utc)
    t1 = t0 + timedelta(seconds=1)

    d0 = _create_detection(frame_id=0, dt=t0)
    d1 = _create_detection(frame_id=30, dt=t1)

    tracklet = Tracklet(
        track_id=1,
        state=TrackState.TRACKED,
        current_detection=d1,
        first_seen_timestamp=t0,
        last_seen_timestamp=t1,
        observation_history=(d0, d1),
    )

    assert tracklet.track_id == 1
    assert tracklet.state == TrackState.TRACKED
    assert tracklet.observation_count == 2
    assert tracklet.lifespan_seconds == pytest.approx(1.0)
    assert tracklet.current_detection == d1


def test_reject_negative_track_id():
    now = datetime.now(timezone.utc)
    d = _create_detection(frame_id=0, dt=now)

    with pytest.raises(ValidationError):
        Tracklet(
            track_id=-1,
            state=TrackState.NEW,
            current_detection=d,
            first_seen_timestamp=now,
            last_seen_timestamp=now,
        )


def test_reject_last_seen_preceding_first_seen():
    t0 = datetime(2026, 9, 27, 12, 0, 10, tzinfo=timezone.utc)
    t_invalid = datetime(2026, 9, 27, 12, 0, 0, tzinfo=timezone.utc)
    d = _create_detection(frame_id=0, dt=t0)

    with pytest.raises(ValidationError, match="cannot precede"):
        Tracklet(
            track_id=2,
            state=TrackState.NEW,
            current_detection=d,
            first_seen_timestamp=t0,
            last_seen_timestamp=t_invalid,
        )


def test_tracklet_lifecycle_states():
    now = datetime.now(timezone.utc)
    d = _create_detection(frame_id=0, dt=now)

    for state in (TrackState.NEW, TrackState.TRACKED, TrackState.COASTING, TrackState.LOST):
        trk = Tracklet(
            track_id=10,
            state=state,
            current_detection=d,
            first_seen_timestamp=now,
            last_seen_timestamp=now,
        )
        assert trk.state == state


def test_tracklet_immutability():
    now = datetime.now(timezone.utc)
    d = _create_detection(frame_id=0, dt=now)
    tracklet = Tracklet(
        track_id=5,
        state=TrackState.NEW,
        current_detection=d,
        first_seen_timestamp=now,
        last_seen_timestamp=now,
    )
    with pytest.raises(ValidationError):
        tracklet.state = TrackState.TRACKED
