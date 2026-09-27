"""Data contracts for structured incident alerts dispatched by perception system."""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, ConfigDict, Field

from src.domain.enums import AlertSeverity, IncidentStatus, TargetClass
from src.schemas.detection import BoundingBox
from src.schemas.telemetry import UavTelemetry


class IncidentAlert(BaseModel):
    """Immutable structured incident alert emitted by the perception pipeline.

    Represents a verified or candidate event (such as a person in water or drowning distress)
    ready for consumption by an autonomous flight controller, mission planner, or ground station.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    incident_id: str = Field(
        ...,
        description="Globally unique identifier for this incident event",
    )
    schema_version: str = Field(
        default="1.0.0",
        description="Semantic version of the incident alert schema",
    )
    timestamp: datetime = Field(
        ...,
        description="UTC timestamp when the incident alert was generated",
    )
    frame_id: int = Field(
        ...,
        ge=0,
        description="Zero-indexed monotonic video frame identifier",
    )
    video_time_sec: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Elapsed playback time in video stream (seconds)",
    )
    incident_status: IncidentStatus = Field(
        ...,
        description="Lifecycle status of the incident (DETECTED, CANDIDATE, CONFIRMED, etc.)",
    )
    target_class: TargetClass = Field(
        ...,
        description="Canonical classification of the detected target entity",
    )
    severity: AlertSeverity = Field(
        ...,
        description="Operational urgency level (INFO, LOW, MEDIUM, HIGH, CRITICAL)",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Aggregated confidence score in range [0.0, 1.0]",
    )
    track_id: Optional[int] = Field(
        default=None,
        ge=0,
        description="Associated tracker target identity if tracking is active",
    )
    bbox: Optional[BoundingBox] = Field(
        default=None,
        description="Current pixel bounding box of target in frame",
    )
    telemetry: Optional[UavTelemetry] = Field(
        default=None,
        description="Drone telemetry snapshot synchronized with this observation",
    )
    recommended_action: Optional[str] = Field(
        default=None,
        description="Suggested action for mission planner (e.g. HOVER_AND_MAINTAIN_OBSERVATION)",
    )
