"""Unit tests for incident alert data contracts."""

from datetime import datetime, timezone
import pytest
from pydantic import ValidationError

from src.domain.enums import AlertSeverity, IncidentStatus, TargetClass
from src.schemas.detection import BoundingBox
from src.schemas.incident import IncidentAlert
from src.schemas.telemetry import UavTelemetry


def test_valid_incident_alert():
    now = datetime.now(timezone.utc)
    bbox = BoundingBox(x1=200.0, y1=150.0, x2=240.0, y2=190.0)
    telem = UavTelemetry(
        timestamp=now,
        altitude_agl_m=30.0,
        gimbal_pitch_deg=-60.0,
    )

    alert = IncidentAlert(
        incident_id="inc-sar-001",
        schema_version="1.0.0",
        timestamp=now,
        frame_id=150,
        video_time_sec=5.0,
        incident_status=IncidentStatus.CONFIRMED,
        target_class=TargetClass.SWIMMER,
        severity=AlertSeverity.CRITICAL,
        confidence=0.92,
        track_id=7,
        bbox=bbox,
        telemetry=telem,
        recommended_action="DISPATCH_RESCUE_BOAT",
    )

    assert alert.incident_id == "inc-sar-001"
    assert alert.incident_status == IncidentStatus.CONFIRMED
    assert alert.target_class == TargetClass.SWIMMER
    assert alert.severity == AlertSeverity.CRITICAL
    assert alert.confidence == 0.92
    assert alert.track_id == 7
    assert alert.frame_id == 150
    assert alert.video_time_sec == 5.0
    assert alert.recommended_action == "DISPATCH_RESCUE_BOAT"


def test_reject_invalid_confidence():
    now = datetime.now(timezone.utc)
    with pytest.raises(ValidationError):
        IncidentAlert(
            incident_id="inc-002",
            timestamp=now,
            frame_id=1,
            incident_status=IncidentStatus.DETECTED,
            target_class=TargetClass.PERSON_SURFACE,
            severity=AlertSeverity.INFO,
            confidence=1.05,  # > 1.0
        )

    with pytest.raises(ValidationError):
        IncidentAlert(
            incident_id="inc-003",
            timestamp=now,
            frame_id=1,
            incident_status=IncidentStatus.DETECTED,
            target_class=TargetClass.PERSON_SURFACE,
            severity=AlertSeverity.INFO,
            confidence=-0.1,  # < 0.0
        )


def test_reject_negative_frame_or_video_time():
    now = datetime.now(timezone.utc)
    with pytest.raises(ValidationError):
        IncidentAlert(
            incident_id="inc-004",
            timestamp=now,
            frame_id=-1,  # negative frame
            incident_status=IncidentStatus.DETECTED,
            target_class=TargetClass.FLOATER,
            severity=AlertSeverity.LOW,
            confidence=0.5,
        )

    with pytest.raises(ValidationError):
        IncidentAlert(
            incident_id="inc-005",
            timestamp=now,
            frame_id=0,
            video_time_sec=-0.5,  # negative time
            incident_status=IncidentStatus.DETECTED,
            target_class=TargetClass.FLOATER,
            severity=AlertSeverity.LOW,
            confidence=0.5,
        )


def test_reject_invalid_enums():
    now = datetime.now(timezone.utc)
    with pytest.raises(ValidationError):
        IncidentAlert(
            incident_id="inc-006",
            timestamp=now,
            frame_id=0,
            incident_status="INVALID_STATUS",  # invalid enum
            target_class=TargetClass.SWIMMER,
            severity=AlertSeverity.HIGH,
            confidence=0.8,
        )
