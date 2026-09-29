"""Pure temporal distress behavior analysis and multi-feature classification."""

from datetime import datetime
import math
from typing import Dict, Optional

from src.analysis.temporal_features import (
    extract_temporal_features,
    slice_tracklet_window,
)
from src.domain.enums import BehaviorState, TargetClass
from src.schemas.behavior import (
    BehaviorAssessment,
    BehaviorClassifierConfig,
    BehaviorEvidence,
)
from src.schemas.temporal_features import TemporalFeatureConfig, TemporalFeatures
from src.schemas.tracking import Tracklet


class _TrackBehaviorHistory:
    """Internal tracker state for smoothing and confirming temporal behavior."""

    def __init__(self) -> None:
        self.state: BehaviorState = BehaviorState.UNKNOWN
        self.consecutive_distress_windows: int = 0
        self.consecutive_normal_windows: int = 0
        self.last_assessment: Optional[BehaviorAssessment] = None


class TemporalDistressClassifier:
    """Detector-agnostic classifier for analyzing swimmer temporal dynamics.

    Evaluates rolling temporal windows of Tracklets to distinguish between
    calm/directed swimming (NORMAL) and erratic/ineffective struggling
    (DISTRESS_CANDIDATE, DISTRESS_CONFIRMED).

    Uses a multi-feature explainable baseline combining:
    1. Locomotion efficiency (tortuosity / straightness ratio)
    2. Directional volatility (deflections per second)
    3. Bounding-box scale growth (splashing / surface submersion)
    4. Detection gap density (intermittent wave submergence)
    5. Kinetic volatility (speed variation vs median progress)
    """

    def __init__(self, config: Optional[BehaviorClassifierConfig] = None) -> None:
        self.config = config or BehaviorClassifierConfig()
        self._track_histories: Dict[int, _TrackBehaviorHistory] = {}

    def reset(self, track_id: Optional[int] = None) -> None:
        """Reset historical behavior state for a specific track or all tracks."""
        if track_id is not None:
            self._track_histories.pop(track_id, None)
        else:
            self._track_histories.clear()

    def get_assessment(self, track_id: int) -> Optional[BehaviorAssessment]:
        """Retrieve latest behavior assessment for a track if available."""
        history = self._track_histories.get(track_id)
        return history.last_assessment if history else None

    def evaluate_features(self, features: TemporalFeatures) -> tuple[float, BehaviorEvidence]:
        """Compute an explainable distress score and component evidence from temporal features.

        Args:
            features: Extracted temporal features over the evaluation window.

        Returns:
            Tuple of (overall_distress_score [0.0..1.0], BehaviorEvidence breakdown).
        """
        cfg = self.config

        # 1. Locomotion efficiency (low straightness indicates churning in place)
        path_len = features.total_path_length_px
        disp = features.total_displacement_px
        straightness = disp / max(1.0, path_len)

        if path_len < cfg.min_path_length_px:
            # Virtually no motion: neutral calm in place
            locomotion_score = 0.0
        elif straightness <= cfg.max_straightness_distress:
            # Low straightness with notable path length = struggling / circling
            locomotion_score = 1.0 - (straightness / max(1e-4, cfg.max_straightness_distress))
            locomotion_score = max(0.0, min(1.0, locomotion_score))
        else:
            # Directed translational swimming
            locomotion_score = 0.0

        # 2. Directional volatility (deflections / sec)
        duration = max(0.1, features.duration_seconds)
        dir_rate = features.direction_change_count / duration
        direction_score = max(0.0, min(1.0, dir_rate / max(1e-4, cfg.distress_direction_rate_hz)))

        # 3. Bounding-box area growth / scale fluctuation (splashing)
        growth = features.bbox_growth_ratio
        if growth >= cfg.distress_growth_ratio:
            growth_excess = (growth - cfg.distress_growth_ratio) / max(0.5, cfg.distress_growth_ratio)
            growth_score = max(0.0, min(1.0, 0.5 + 0.5 * growth_excess))
        else:
            growth_score = max(0.0, min(0.5, (growth - 1.0) / max(0.5, cfg.distress_growth_ratio - 1.0) * 0.5))

        # 4. Submersion / gap frequency
        total_f = max(1, features.total_frames)
        gap_frames = total_f - features.observation_count
        gap_density = gap_frames / total_f
        submersion_score = max(0.0, min(1.0, gap_density / max(1e-4, cfg.min_gap_density * 2.0)))

        # 5. Kinetic volatility (speed deviation relative to median speed)
        med_spd = features.median_speed_px_per_sec
        max_spd = features.max_speed_px_per_sec
        mean_spd = features.mean_speed_px_per_sec
        accel = features.mean_acceleration_px_per_sec2

        # Check if acceleration and speed spread are elevated
        spd_spread = max(0.0, max_spd - med_spd)
        spread_ratio = spd_spread / max(10.0, med_spd + mean_spd)
        accel_ratio = accel / max(20.0, mean_spd * 2.0)
        kinetic_score = max(0.0, min(1.0, 0.5 * spread_ratio + 0.5 * accel_ratio))

        # Weighted combination
        raw_score = (
            cfg.weight_locomotion_inefficiency * locomotion_score
            + cfg.weight_directional_volatility * direction_score
            + cfg.weight_scale_fluctuation * growth_score
            + cfg.weight_submersion_gaps * submersion_score
            + cfg.weight_kinetic_volatility * kinetic_score
        )
        distress_score = round(max(0.0, min(1.0, raw_score)), 3)

        evidence = BehaviorEvidence(
            locomotion_score=round(locomotion_score, 3),
            direction_score=round(direction_score, 3),
            growth_score=round(growth_score, 3),
            submersion_score=round(submersion_score, 3),
            kinetic_score=round(kinetic_score, 3),
            straightness_ratio=round(straightness, 3),
            direction_change_rate=round(dir_rate, 2),
            bbox_growth_ratio=round(growth, 2),
            gap_density=round(gap_density, 3),
            median_speed_px_per_sec=round(med_spd, 2),
        )
        return distress_score, evidence

    def update(
        self,
        tracklet: Tracklet,
        default_fps: float = 10.0,
        current_timestamp: Optional[datetime] = None,
    ) -> BehaviorAssessment:
        """Evaluate a tracked target's temporal window and return its behavioral assessment.

        Args:
            tracklet: Full-history domain Tracklet.
            default_fps: Video frame rate.
            current_timestamp: Optional reference timestamp.

        Returns:
            Validated immutable BehaviorAssessment object.
        """
        tid = tracklet.track_id
        target_cls = tracklet.current_detection.target_class

        if tid not in self._track_histories:
            self._track_histories[tid] = _TrackBehaviorHistory()
        history = self._track_histories[tid]

        # Slice rolling temporal window
        window_trk = slice_tracklet_window(
            tracklet=tracklet,
            window_seconds=self.config.window_seconds,
            end_timestamp=current_timestamp,
            default_fps=default_fps,
        )

        # Extract features over the window
        feat_cfg = TemporalFeatureConfig(default_fps=default_fps)
        window_features = extract_temporal_features(window_trk, config=feat_cfg)

        obs_count = window_features.observation_count
        duration = window_features.duration_seconds

        # Non-swimmer targets: immediately classify as NORMAL (not at distress risk)
        if target_cls not in (TargetClass.SWIMMER, TargetClass.PERSON, TargetClass.PERSON_SURFACE):
            assessment = BehaviorAssessment(
                track_id=tid,
                target_class=target_cls,
                state=BehaviorState.NORMAL,
                distress_score=0.0,
                confidence=1.0,
                window_duration_seconds=duration,
                observation_count=obs_count,
                consecutive_distress_windows=0,
                consecutive_normal_windows=history.consecutive_normal_windows + 1,
                evidence=None,
                explanation=f"Target class '{target_cls.value}' is exempt from distress evaluation.",
            )
            history.last_assessment = assessment
            return assessment

        # Check for sufficient temporal evidence in window
        if obs_count < self.config.min_observations or duration < self.config.min_window_duration_seconds:
            # Low confidence due to sparse observations
            temporal_confidence = round(min(0.4, obs_count / max(1, self.config.min_observations) * 0.4), 2)
            assessment = BehaviorAssessment(
                track_id=tid,
                target_class=target_cls,
                state=BehaviorState.UNKNOWN,
                distress_score=0.0,
                confidence=temporal_confidence,
                window_duration_seconds=duration,
                observation_count=obs_count,
                consecutive_distress_windows=history.consecutive_distress_windows,
                consecutive_normal_windows=history.consecutive_normal_windows,
                evidence=None,
                explanation=f"Insufficient temporal observations ({obs_count}/{self.config.min_observations} in {duration:.2f}s).",
            )
            history.last_assessment = assessment
            return assessment

        # Evaluate multi-feature evidence
        distress_score, evidence = self.evaluate_features(window_features)

        # Behavioral confidence increases with observation density and duration stability
        temporal_confidence = round(min(1.0, 0.6 + 0.4 * (obs_count / (self.config.min_observations * 2))), 2)

        # State transition logic with temporal smoothing
        if distress_score >= self.config.distress_threshold:
            history.consecutive_distress_windows += 1
            history.consecutive_normal_windows = 0

            if history.consecutive_distress_windows >= self.config.confirm_windows:
                history.state = BehaviorState.DISTRESS_CONFIRMED
                explanation = (
                    f"Persistent distress indicators across {history.consecutive_distress_windows} windows: "
                    f"inefficiency={evidence.locomotion_score:.2f}, direction_rate={evidence.direction_change_rate:.1f}Hz, "
                    f"growth={evidence.bbox_growth_ratio:.1f}x."
                )
            else:
                history.state = BehaviorState.DISTRESS_CANDIDATE
                explanation = (
                    f"Distress candidate detected ({history.consecutive_distress_windows}/{self.config.confirm_windows} windows): "
                    f"score={distress_score:.2f} >= {self.config.distress_threshold:.2f}."
                )
        else:
            history.consecutive_normal_windows += 1
            history.consecutive_distress_windows = 0

            # If previously confirmed distress, require recovery_windows to de-escalate
            if history.state == BehaviorState.DISTRESS_CONFIRMED:
                if history.consecutive_normal_windows >= self.config.recovery_windows:
                    history.state = BehaviorState.NORMAL
                    explanation = f"Distress resolved after {history.consecutive_normal_windows} consecutive normal windows."
                else:
                    # Maintain confirmed distress through momentary pause
                    explanation = (
                        f"Maintaining distress confirmation ({history.consecutive_normal_windows}/{self.config.recovery_windows} "
                        f"recovery windows, current score={distress_score:.2f})."
                    )
            else:
                history.state = BehaviorState.NORMAL
                explanation = (
                    f"Normal swimming behavior: straightness={evidence.straightness_ratio:.2f}, "
                    f"dir_rate={evidence.direction_change_rate:.1f}Hz, score={distress_score:.2f} < {self.config.distress_threshold:.2f}."
                )

        assessment = BehaviorAssessment(
            track_id=tid,
            target_class=target_cls,
            state=history.state,
            distress_score=distress_score,
            confidence=temporal_confidence,
            window_duration_seconds=duration,
            observation_count=obs_count,
            consecutive_distress_windows=history.consecutive_distress_windows,
            consecutive_normal_windows=history.consecutive_normal_windows,
            evidence=evidence,
            explanation=explanation,
        )
        history.last_assessment = assessment
        return assessment
