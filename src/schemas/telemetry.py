"""Data contracts for aerial drone telemetry and camera gimbal metadata."""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, ConfigDict, Field, model_validator


class UavTelemetry(BaseModel):
    """Immutable snapshot of drone platform position, attitude, and gimbal state.

    All geographic and angular fields are optional to support operation in
    GPS-denied, uncalibrated, or simulated environments without breaking pipeline contracts.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    timestamp: datetime = Field(
        ...,
        description="Measurement timestamp in UTC",
    )
    altitude_agl_m: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=10000.0,
        description="Altitude above ground level in meters (non-negative)",
    )
    latitude: Optional[float] = Field(
        default=None,
        ge=-90.0,
        le=90.0,
        description="WGS84 latitude in degrees (-90 to +90)",
    )
    longitude: Optional[float] = Field(
        default=None,
        ge=-180.0,
        le=180.0,
        description="WGS84 longitude in degrees (-180 to +180)",
    )
    heading_deg: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=360.0,
        description="Compass heading in degrees clockwise from True North (0 to 360)",
    )
    gimbal_pitch_deg: Optional[float] = Field(
        default=None,
        ge=-90.0,
        le=90.0,
        description="Camera gimbal pitch angle in degrees (-90 nadir to +90 zenith)",
    )
    gps_fix_available: bool = Field(
        default=False,
        description="True if reliable navigation satellite fix is locked",
    )

    @model_validator(mode="after")
    def validate_coordinate_pair_and_gps(self) -> "UavTelemetry":
        has_lat = self.latitude is not None
        has_lon = self.longitude is not None

        if has_lat != has_lon:
            raise ValueError(
                "Both latitude and longitude must be provided together or both omitted."
            )

        if self.gps_fix_available and not (has_lat and has_lon):
            raise ValueError(
                "gps_fix_available is True but valid latitude and longitude coordinates were not provided."
            )

        return self
