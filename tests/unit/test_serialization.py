"""Unit tests for schema serialization and JSON round-trip fidelity."""

from datetime import datetime, timezone
import pytest

from src.domain.enums import AlertSeverity, IncidentStatus, TargetClass, TrackState
from src.schemas.detection import BoundingBox, Detection
from src.schemas.incident import IncidentAlert
from src.schemas.serialization import from_dict, from_json, to_dict, to_json
from src.schemas.telemetry import UavTelemetry
from src.schemas.tracking import Tracklet


def test_detection_json_round_trip():
    now = datetime(2026, 9, 27, 15, 30, 0, tzinfo=timezone.utc)
    det = Detection(
        bbox=BoundingBox(x1=15.5, y1=25.5, x2=85.0, y2=95.0),
        target_class=TargetClass.SWIMMER,
        confidence=0.912,
        frame_id=120,
        timestamp=now,
        detector_name="baseline_yolov8n",
    )

    json_str = to_json(det)
    assert isinstance(json_str, str)
    assert "swimmer" in json_str

    restored = from_json(Detection, json_str)
    assert restored == det
    assert restored.bbox.width == pytest.approx(69.5)
    assert restored.confidence == pytest.approx(0.912)


def test_tracklet_dict_round_trip():
    now = datetime(2026, 9, 27, 15, 30, 0, tzinfo=timezone.utc)
    det = Detection(
        bbox=BoundingBox(x1=10.0, y1=10.0, x2=40.0, y2=40.0),
        target_class=TargetClass.FLOATER,
        confidence=0.85,
        frame_id=10,
        timestamp=now,
    )
    tracklet = Tracklet(
        track_id=4,
        state=TrackState.TRACKED,
        current_detection=det,
        first_seen_timestamp=now,
        last_seen_timestamp=now,
        observation_history=(det,),
    )

    data = to_dict(tracklet)
    assert isinstance(data, dict)
    assert data["track_id"] == 4
    assert data["state"] == "tracked"

    restored = from_dict(Tracklet, data)
    assert restored == tracklet
    assert restored.observation_count == 1


def test_incident_alert_json_round_trip():
    now = datetime(2026, 9, 27, 15, 30, 0, tzinfo=timezone.utc)
    alert = IncidentAlert(
        incident_id="inc-uuid-12345",
        schema_version="1.0.0",
        timestamp=now,
        frame_id=300,
        video_time_sec=10.0,
        incident_status=IncidentStatus.CANDIDATE,
        target_class=TargetClass.PERSON_SURFACE,
        severity=AlertSeverity.HIGH,
        confidence=0.88,
        track_id=9,
        bbox=BoundingBox(x1=50.0, y1=60.0, x2=100.0, y2=120.0),
        telemetry=UavTelemetry(
            timestamp=now,
            altitude_agl_m=25.0,
            gimbal_pitch_deg=-45.0,
        ),
        recommended_action="APPROACH_FOR_CONFIRMATION",
    )

    json_str = to_json(alert, indent=2)
    assert "inc-uuid-12345" in json_str
    assert "candidate" in json_str

    restored = from_json(IncidentAlert, json_str)
    assert restored.incident_id == alert.incident_id
    assert restored.incident_status == alert.incident_status
    assert restored.target_class == alert.target_class
    assert restored.severity == alert.severity
    assert restored.confidence == pytest.approx(alert.confidence)
    assert restored.telemetry.altitude_agl_m == pytest.approx(25.0)


def test_reject_malformed_json_strings():
    with pytest.raises(ValueError, match="Failed to validate JSON"):
        from_json(Detection, "not valid json {")

    with pytest.raises(ValueError, match="Cannot parse empty"):
        from_json(Detection, "")

    with pytest.raises(ValueError, match="Cannot parse empty"):
        from_json(Detection, "   ")


def test_reject_json_with_schema_violations():
    # Negative coordinate in JSON
    invalid_json = """
    {
        "bbox": {"x1": -10.0, "y1": 10.0, "x2": 50.0, "y2": 50.0},
        "target_class": "swimmer",
        "confidence": 0.8,
        "frame_id": 1,
        "timestamp": "2026-09-27T12:00:00Z"
    }
    """
    with pytest.raises(ValueError, match="Failed to validate JSON"):
        from_json(Detection, invalid_json)

    # Confidence > 1.0 in JSON
    invalid_conf_json = """
    {
        "bbox": {"x1": 10.0, "y1": 10.0, "x2": 50.0, "y2": 50.0},
        "target_class": "swimmer",
        "confidence": 1.5,
        "frame_id": 1,
        "timestamp": "2026-09-27T12:00:00Z"
    }
    """
    with pytest.raises(ValueError, match="Failed to validate JSON"):
        from_json(Detection, invalid_conf_json)
