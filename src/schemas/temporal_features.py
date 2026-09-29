"""Data contracts for track diagnostics and temporal feature extraction."""

from typing import Optional, Tuple
from pydantic import BaseModel, ConfigDict, Field

from src.domain.enums import TargetClass


class TemporalFeatureConfig(BaseModel):
    """Configuration parameters for extracting temporal features from Tracklet histories."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    default_fps: float = Field(
        default=10.0,
        gt=0.0,
        description="Fallback video frame rate (FPS) used when timestamps are identical or missing",
    )
    min_movement_px: float = Field(
        default=2.0,
        ge=0.0,
        description="Displacement threshold (pixels) below which motion vectors are discarded as jitter",
    )
    direction_change_threshold_deg: float = Field(
        default=45.0,
        ge=0.0,
        le=180.0,
        description="Angular threshold (degrees) between consecutive motion vectors to count as direction change",
    )
    coasting_detector_name: str = Field(
        default="kalman_coasting",
        description="Identifier used to distinguish synthetic Kalman coasting predictions from real detections",
    )


class TemporalFeatures(BaseModel):
    """Immutable representation of extracted temporal dynamics and kinematics for a Tracklet.

    IMPORTANT: All spatial measurements are strictly image-space pixel units.
    They do NOT represent physical-world metrics (e.g. meters or m/s) because
    camera intrinsics/extrinsics, lens distortion, and UAV altitude/attitude
    are not calibrated at this stage.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    track_id: int = Field(
        ...,
        ge=0,
        description="Unique non-negative target identity assigned by the tracker",
    )
    target_class: TargetClass = Field(
        ...,
        description="Semantic target class associated with the tracklet",
    )

    # Lifespan, observations, and gap metrics
    duration_seconds: float = Field(
        ...,
        ge=0.0,
        description="Total elapsed temporal duration of the track in seconds",
    )
    observation_count: int = Field(
        ...,
        ge=0,
        description="Total count of verified detector observations (excluding coasting/predictions)",
    )
    total_frames: int = Field(
        ...,
        ge=0,
        description="Total frame span between first and last observed/coasted frame",
    )
    visibility_ratio: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Ratio of verified detector observations to total frame lifespan",
    )
    detection_gap_count: int = Field(
        ...,
        ge=0,
        description="Number of distinct detection gap occurrences where the target was missed",
    )
    max_consecutive_gap: int = Field(
        ...,
        ge=0,
        description="Maximum number of consecutive frames missed in a single gap event",
    )

    # Spatial centroid & displacement metrics (image-space pixels)
    start_centroid: Optional[Tuple[float, float]] = Field(
        default=None,
        description="Starting (x, y) centroid in pixel coordinates",
    )
    end_centroid: Optional[Tuple[float, float]] = Field(
        default=None,
        description="Ending (x, y) centroid in pixel coordinates",
    )
    total_displacement_px: float = Field(
        ...,
        ge=0.0,
        description="Net Euclidean distance between starting and ending centroid (pixels)",
    )
    total_path_length_px: float = Field(
        ...,
        ge=0.0,
        description="Cumulative Euclidean path distance traversed across consecutive observations (pixels)",
    )

    # Kinematics (image-space pixel derivatives)
    mean_speed_px_per_sec: float = Field(
        ...,
        ge=0.0,
        description="Mean image-space speed across consecutive observation intervals (pixels/second)",
    )
    max_speed_px_per_sec: float = Field(
        ...,
        ge=0.0,
        description="Peak image-space speed observed across any single interval (pixels/second)",
    )
    mean_acceleration_px_per_sec2: float = Field(
        ...,
        ge=0.0,
        description="Mean image-space acceleration magnitude across consecutive intervals (pixels/second^2)",
    )
    direction_change_count: int = Field(
        ...,
        ge=0,
        description="Number of significant directional changes exceeding the angular threshold",
    )

    # Bounding-box scale and growth
    min_bbox_area_px2: float = Field(
        ...,
        ge=0.0,
        description="Minimum observed bounding-box area in pixel units (width * height)",
    )
    max_bbox_area_px2: float = Field(
        ...,
        ge=0.0,
        description="Maximum observed bounding-box area in pixel units (width * height)",
    )
    bbox_growth_ratio: float = Field(
        ...,
        ge=0.0,
        description="Ratio of maximum to minimum observed bounding-box area",
    )

    # Detection quality
    mean_confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Mean confidence score across verified detector observations",
    )
