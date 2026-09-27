"""Unit tests for drone telemetry data contracts."""

from datetime import datetime, timezone
import pytest
from pydantic import ValidationError

from src.schemas.telemetry import UavTelemetry


def test_valid_telemetry_with_full_context():
    now = datetime.now(timezone.utc)
    telem = UavTelemetry(
        timestamp=now,
        altitude_agl_m=35.5,
        latitude=47.6062,
        longitude=-122.3321,
        heading_deg=180.0,
        gimbal_pitch_deg=-45.0,
        gps_fix_available=True,
    )
    assert telem.timestamp == now
    assert telem.altitude_agl_m == 35.5
    assert telem.latitude == 47.6062
    assert telem.longitude == -122.3321
    assert telem.heading_deg == 180.0
    assert telem.gimbal_pitch_deg == -45.0
    assert telem.gps_fix_available is True


def test_valid_telemetry_without_gps():
    now = datetime.now(timezone.utc)
    telem = UavTelemetry(
        timestamp=now,
        altitude_agl_m=20.0,
        gimbal_pitch_deg=-90.0,
        gps_fix_available=False,
    )
    assert telem.latitude is None
    assert telem.longitude is None
    assert telem.gps_fix_available is False
    assert telem.altitude_agl_m == 20.0


def test_reject_gps_fix_without_coordinates():
    now = datetime.now(timezone.utc)
    with pytest.raises(ValidationError, match="gps_fix_available is True but valid latitude"):
        UavTelemetry(
            timestamp=now,
            gps_fix_available=True,
        )


def test_reject_partial_coordinates():
    now = datetime.now(timezone.utc)
    with pytest.raises(ValidationError, match="Both latitude and longitude must be provided"):
        UavTelemetry(
            timestamp=now,
            latitude=45.0,
            longitude=None,
        )

    with pytest.raises(ValidationError, match="Both latitude and longitude must be provided"):
        UavTelemetry(
            timestamp=now,
            latitude=None,
            longitude=10.0,
        )


def test_reject_out_of_bounds_physical_values():
    now = datetime.now(timezone.utc)

    # Negative altitude
    with pytest.raises(ValidationError):
        UavTelemetry(timestamp=now, altitude_agl_m=-1.0)

    # Latitude out of bounds
    with pytest.raises(ValidationError):
        UavTelemetry(timestamp=now, latitude=91.0, longitude=0.0)

    with pytest.raises(ValidationError):
        UavTelemetry(timestamp=now, latitude=-91.0, longitude=0.0)

    # Longitude out of bounds
    with pytest.raises(ValidationError):
        UavTelemetry(timestamp=now, latitude=0.0, longitude=181.0)

    # Heading out of [0, 360]
    with pytest.raises(ValidationError):
        UavTelemetry(timestamp=now, heading_deg=361.0)

    with pytest.raises(ValidationError):
        UavTelemetry(timestamp=now, heading_deg=-5.0)

    # Gimbal pitch out of [-90, 90]
    with pytest.raises(ValidationError):
        UavTelemetry(timestamp=now, gimbal_pitch_deg=-95.0)

    with pytest.raises(ValidationError):
        UavTelemetry(timestamp=now, gimbal_pitch_deg=95.0)
