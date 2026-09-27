"""Data contracts for single-frame spatial detections."""

from datetime import datetime
from typing import Optional, Tuple
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.domain.enums import TargetClass


class BoundingBox(BaseModel):
    """Immutable pixel-space bounding box representation in xyxy format.

    Invariants:
        x1 >= 0.0
        y1 >= 0.0
        x2 > x1
        y2 > y1
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    x1: float = Field(..., description="Top-left corner x coordinate (pixels)")
    y1: float = Field(..., description="Top-left corner y coordinate (pixels)")
    x2: float = Field(..., description="Bottom-right corner x coordinate (pixels)")
    y2: float = Field(..., description="Bottom-right corner y coordinate (pixels)")

    @model_validator(mode="after")
    def validate_box_invariants(self) -> "BoundingBox":
        if self.x1 < 0.0 or self.y1 < 0.0:
            raise ValueError(
                f"Bounding box coordinates must be non-negative: (x1={self.x1}, y1={self.y1})"
            )
        if self.x2 <= self.x1:
            raise ValueError(
                f"Invalid box width: x2 ({self.x2}) must be strictly greater than x1 ({self.x1})"
            )
        if self.y2 <= self.y1:
            raise ValueError(
                f"Invalid box height: y2 ({self.y2}) must be strictly greater than y1 ({self.y1})"
            )
        return self

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> Tuple[float, float]:
        return (self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0

    @property
    def aspect_ratio(self) -> float:
        """Width to height aspect ratio."""
        return self.width / self.height

    def as_xyxy(self) -> Tuple[float, float, float, float]:
        return self.x1, self.y1, self.x2, self.y2

    def as_xywh(self) -> Tuple[float, float, float, float]:
        return self.x1, self.y1, self.width, self.height


class Detection(BaseModel):
    """Immutable single-frame spatial detection output."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    bbox: BoundingBox
    target_class: TargetClass
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Model confidence score bounded between 0.0 and 1.0",
    )
    frame_id: int = Field(
        ...,
        ge=0,
        description="Zero-indexed monotonic video frame counter",
    )
    timestamp: datetime = Field(
        ...,
        description="Observation timestamp (UTC timezone recommended)",
    )
    detector_name: Optional[str] = Field(
        default=None,
        description="Optional identifier of the generating detector model",
    )
