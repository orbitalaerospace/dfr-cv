"""Data contracts for temporal behavior analysis and distress classification."""

from typing import Dict, Optional
from pydantic import BaseModel, ConfigDict, Field

from src.domain.enums import BehaviorState, TargetClass


class BehaviorClassifierConfig(BaseModel):
    """Configuration parameters for temporal distress behavior analysis.

    Defines windowing, feature thresholds, weights, and confirmation parameters.
    No unexplained magic numbers are hardcoded in the classifier logic.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    # Rolling window parameters
    window_seconds: float = Field(
        default=2.0,
        gt=0.0,
        description="Temporal duration (seconds) of the rolling observation window",
    )
    stride_seconds: float = Field(
        default=0.5,
        gt=0.0,
        description="Temporal stride (seconds) between successive window evaluations",
    )
    min_observations: int = Field(
        default=5,
        ge=2,
        description="Minimum number of verified detections required before evaluating behavior",
    )
    min_window_duration_seconds: float = Field(
        default=0.3,
        ge=0.1,
        description="Minimum duration in seconds required in window to transition from UNKNOWN",
    )


    # Distress score threshold & confirmation
    distress_threshold: float = Field(
        default=0.55,
        ge=0.0,
        le=1.0,
        description="Instantaneous distress score threshold to qualify as distress evidence",
    )
    confirm_windows: int = Field(
        default=3,
        ge=1,
        description="Consecutive distress windows required to confirm DISTRESS_CONFIRMED",
    )
    recovery_windows: int = Field(
        default=3,
        ge=1,
        description="Consecutive normal windows required to de-escalate back to NORMAL",
    )

    # Feature weights (sum to 1.0)
    weight_locomotion_inefficiency: float = Field(
        default=0.30,
        ge=0.0,
        le=1.0,
        description="Weight for low straightness/displacement relative to path length (churning in place)",
    )
    weight_directional_volatility: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description="Weight for high rate of directional deflections per second",
    )
    weight_scale_fluctuation: float = Field(
        default=0.20,
        ge=0.0,
        le=1.0,
        description="Weight for bounding-box area expansion/contraction (splashing / submerging)",
    )
    weight_submersion_gaps: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="Weight for detection gap frequency / low visibility ratio",
    )
    weight_kinetic_volatility: float = Field(
        default=0.10,
        ge=0.0,
        le=1.0,
        description="Weight for speed/acceleration variation vs mean translational speed",
    )

    # Feature evaluation thresholds
    min_path_length_px: float = Field(
        default=6.0,
        ge=0.0,
        description="Minimum path length in pixels required before evaluating locomotion efficiency",
    )
    max_straightness_distress: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description="Maximum displacement/path ratio indicating inefficient struggling in place",
    )
    distress_direction_rate_hz: float = Field(
        default=1.5,
        ge=0.0,
        description="Direction change frequency (changes/second) indicating erratic panic motion",
    )
    distress_growth_ratio: float = Field(
        default=1.8,
        ge=1.0,
        description="Bounding box max/min area ratio indicating substantial scale fluctuation",
    )
    min_gap_density: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="Ratio of missing gap frames to total window frames indicating intermittent submersion",
    )


class BehaviorEvidence(BaseModel):
    """Component sub-scores and feature measurements for explainable decision support."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    locomotion_score: float = Field(..., ge=0.0, le=1.0, description="Inefficient locomotion sub-score")
    direction_score: float = Field(..., ge=0.0, le=1.0, description="Directional volatility sub-score")
    growth_score: float = Field(..., ge=0.0, le=1.0, description="Scale fluctuation sub-score")
    submersion_score: float = Field(..., ge=0.0, le=1.0, description="Submersion/gap sub-score")
    kinetic_score: float = Field(..., ge=0.0, le=1.0, description="Kinetic volatility sub-score")

    # Raw metrics for auditability
    straightness_ratio: float = Field(..., description="Net displacement / total path length")
    direction_change_rate: float = Field(..., description="Direction changes per second")
    bbox_growth_ratio: float = Field(..., description="Max area / min area")
    gap_density: float = Field(..., description="Gap frames / total frames")
    median_speed_px_per_sec: float = Field(..., description="Median speed across observations")


class BehaviorAssessment(BaseModel):
    """Immutable assessment of a tracked target's temporal behavior.

    IMPORTANT: Distress scores and confidence are empirical image-space heuristic
    measures of behavioral volatility and do NOT represent medically or statistically
    validated drowning probabilities.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    track_id: int = Field(..., ge=0, description="Tracklet identifier")
    target_class: TargetClass = Field(..., description="Semantic object class")
    state: BehaviorState = Field(..., description="Current confirmed or candidate behavioral state")
    distress_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Internal behavioral distress score bounded between 0.0 (calm) and 1.0 (extreme distress)",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Temporal confidence in the behavioral assessment based on observation depth and stability",
    )
    window_duration_seconds: float = Field(..., ge=0.0, description="Duration of observation window evaluated")
    observation_count: int = Field(..., ge=0, description="Number of observations in the evaluation window")
    consecutive_distress_windows: int = Field(..., ge=0, description="Number of consecutive distress-qualifying windows")
    consecutive_normal_windows: int = Field(..., ge=0, description="Number of consecutive normal-qualifying windows")
    evidence: Optional[BehaviorEvidence] = Field(default=None, description="Sub-score breakdown and feature inputs")
    explanation: str = Field(..., description="Human-readable rationale for the behavioral classification")
