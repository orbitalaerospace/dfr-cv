"""Data contracts for object tracking and temporal tracklets."""

from datetime import datetime
from typing import Tuple
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.domain.enums import TrackState
from src.schemas.detection import Detection


class Tracklet(BaseModel):
    """Immutable representation of a tracked object identity across time.

    Encapsulates lifecycle state, current spatial position, and observation
    history without binding to any specific multi-object tracking implementation.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    track_id: int = Field(
        ...,
        ge=0,
        description="Unique non-negative target identity assigned by the tracker",
    )
    state: TrackState = Field(
        ...,
        description="Current lifecycle state (NEW, TRACKED, COASTING, LOST)",
    )
    current_detection: Detection = Field(
        ...,
        description="Most recent spatial observation associated with this identity",
    )
    first_seen_timestamp: datetime = Field(
        ...,
        description="Timestamp when this target was first registered",
    )
    last_seen_timestamp: datetime = Field(
        ...,
        description="Timestamp of the most recent associated detection",
    )
    observation_history: Tuple[Detection, ...] = Field(
        default_factory=tuple,
        description="Chronological record of detections associated with this tracklet",
    )

    @model_validator(mode="after")
    def validate_tracking_invariants(self) -> "Tracklet":
        if self.last_seen_timestamp < self.first_seen_timestamp:
            raise ValueError(
                f"last_seen_timestamp ({self.last_seen_timestamp}) cannot precede "
                f"first_seen_timestamp ({self.first_seen_timestamp})"
            )
        return self

    @property
    def observation_count(self) -> int:
        """Total number of historical observations including current detection."""
        if not self.observation_history:
            return 1
        return len(self.observation_history)

    @property
    def lifespan_seconds(self) -> float:
        """Elapsed time between first registration and latest observation."""
        return (self.last_seen_timestamp - self.first_seen_timestamp).total_seconds()
