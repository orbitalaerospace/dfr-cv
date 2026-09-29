"""Pure analysis module for Tracklet temporal feature extraction and diagnostic reporting."""

from datetime import datetime
import math
from typing import List, Optional, Sequence, Tuple

from src.domain.enums import TargetClass, TrackState
from src.schemas.detection import Detection
from src.schemas.temporal_features import TemporalFeatureConfig, TemporalFeatures
from src.schemas.tracking import Tracklet


def _collect_chronological_detections(tracklet: Tracklet) -> List[Detection]:
    """Assemble all historical and current detections in monotonic chronological order.

    Ensures that current_detection is merged if not already present in observation_history,
    preserving track lifecycle state during both active and coasting frames.
    """
    seen_frame_ids = set()
    observations: List[Detection] = []

    # Process observation history
    for det in tracklet.observation_history:
        if det.frame_id not in seen_frame_ids:
            seen_frame_ids.add(det.frame_id)
            observations.append(det)

    # Include current_detection if not already captured
    if tracklet.current_detection.frame_id not in seen_frame_ids:
        seen_frame_ids.add(tracklet.current_detection.frame_id)
        observations.append(tracklet.current_detection)

    # Sort strictly by frame_id
    observations.sort(key=lambda d: d.frame_id)
    return observations


def extract_temporal_features(
    tracklet: Tracklet,
    config: Optional[TemporalFeatureConfig] = None,
) -> TemporalFeatures:
    """Extract measurable image-space temporal and kinematic features from a Tracklet.

    This function is completely pure:
    Tracklet -> TemporalFeatures.
    It does not depend on OpenCV, YOLO, GPU, or filesystem operations.

    Args:
        tracklet: Domain Tracklet containing lifecycle state and observation history.
        config: Feature extraction configuration parameters (tolerances, thresholds, FPS).

    Returns:
        Immutable, validated TemporalFeatures object.
    """
    cfg = config or TemporalFeatureConfig()

    all_detections = _collect_chronological_detections(tracklet)

    # Partition real vision detector observations from synthetic Kalman coasting predictions
    real_detections = [
        d for d in all_detections
        if d.detector_name != cfg.coasting_detector_name
    ]

    target_cls = tracklet.current_detection.target_class
    track_id = tracklet.track_id

    # If completely empty of detections (defensive check)
    if not all_detections:
        return TemporalFeatures(
            track_id=track_id,
            target_class=target_cls,
            duration_seconds=0.0,
            observation_count=0,
            total_frames=0,
            visibility_ratio=0.0,
            detection_gap_count=0,
            max_consecutive_gap=0,
            start_centroid=None,
            end_centroid=None,
            total_displacement_px=0.0,
            total_path_length_px=0.0,
            mean_speed_px_per_sec=0.0,
            max_speed_px_per_sec=0.0,
            mean_acceleration_px_per_sec2=0.0,
            direction_change_count=0,
            min_bbox_area_px2=0.0,
            max_bbox_area_px2=0.0,
            bbox_growth_ratio=1.0,
            mean_confidence=0.0,
        )

    # 1. Temporal span & gap analysis
    first_frame_id = all_detections[0].frame_id
    last_frame_id = all_detections[-1].frame_id
    total_frames = max(1, last_frame_id - first_frame_id + 1)

    observation_count = len(real_detections)
    visibility_ratio = min(1.0, observation_count / total_frames) if total_frames > 0 else 0.0

    # Gap metrics: gaps between consecutive real detections and trailing gaps
    gaps: List[int] = []
    if real_detections:
        real_frame_ids = [d.frame_id for d in real_detections]
        for i in range(len(real_frame_ids) - 1):
            gap_len = real_frame_ids[i + 1] - real_frame_ids[i] - 1
            if gap_len > 0:
                gaps.append(gap_len)

        # Check if currently coasting after the last real detection
        if last_frame_id > real_frame_ids[-1]:
            trailing_gap = last_frame_id - real_frame_ids[-1]
            if trailing_gap > 0:
                gaps.append(trailing_gap)

    detection_gap_count = len(gaps)
    max_consecutive_gap = max(gaps) if gaps else 0

    # 2. Duration calculation
    # Prefer exact timestamp difference if valid; fallback to frame count and FPS
    t_start = all_detections[0].timestamp
    t_end = all_detections[-1].timestamp
    delta_t_seconds = (t_end - t_start).total_seconds()

    if delta_t_seconds > 0.0:
        duration_seconds = delta_t_seconds
    else:
        duration_seconds = (total_frames - 1) / cfg.default_fps if total_frames > 1 else 0.0

    # 3. Spatial observations to evaluate for kinematics & geometry
    # Use real detections if available; fallback to all detections (e.g. if coasted from start)
    eval_detections = real_detections if real_detections else all_detections

    start_c = eval_detections[0].bbox.center
    end_c = eval_detections[-1].bbox.center
    start_centroid = (round(start_c[0], 2), round(start_c[1], 2))
    end_centroid = (round(end_c[0], 2), round(end_c[1], 2))

    total_displacement_px = math.hypot(
        end_centroid[0] - start_centroid[0],
        end_centroid[1] - start_centroid[1],
    )

    # Bounding-box area metrics
    areas = [d.bbox.area for d in eval_detections]
    min_bbox_area_px2 = min(areas)
    max_bbox_area_px2 = max(areas)
    bbox_growth_ratio = max_bbox_area_px2 / max(1e-6, min_bbox_area_px2)

    # Confidence aggregation across real detector observations only
    if real_detections:
        mean_confidence = sum(d.confidence for d in real_detections) / len(real_detections)
    else:
        mean_confidence = 0.0

    # 4. Kinematic step derivations (distances, speeds, accelerations, direction changes)
    step_distances: List[float] = []
    step_speeds: List[float] = []
    step_dts: List[float] = []
    motion_vectors: List[Tuple[float, float]] = []

    for i in range(len(eval_detections) - 1):
        d_curr = eval_detections[i]
        d_next = eval_detections[i + 1]

        c_curr = d_curr.bbox.center
        c_next = d_next.bbox.center

        dx = c_next[0] - c_curr[0]
        dy = c_next[1] - c_curr[1]
        dist = math.hypot(dx, dy)

        # Time interval between observations
        dt = (d_next.timestamp - d_curr.timestamp).total_seconds()
        if dt <= 0.0:
            frame_diff = d_next.frame_id - d_curr.frame_id
            dt = frame_diff / cfg.default_fps if frame_diff > 0 else (1.0 / cfg.default_fps)

        speed = dist / dt if dt > 0.0 else 0.0

        step_distances.append(dist)
        step_speeds.append(speed)
        step_dts.append(dt)
        motion_vectors.append((dx, dy))

    total_path_length_px = sum(step_distances)

    if step_speeds:
        mean_speed_px_per_sec = sum(step_speeds) / len(step_speeds)
        max_speed_px_per_sec = max(step_speeds)
    else:
        mean_speed_px_per_sec = 0.0
        max_speed_px_per_sec = 0.0

    # Image-space acceleration magnitude: delta_speed / delta_time_midpoint
    if len(step_speeds) >= 2:
        step_accels: List[float] = []
        for i in range(len(step_speeds) - 1):
            dt_acc = 0.5 * (step_dts[i] + step_dts[i + 1])
            if dt_acc > 0.0:
                acc = abs(step_speeds[i + 1] - step_speeds[i]) / dt_acc
                step_accels.append(acc)
        mean_acceleration_px_per_sec2 = sum(step_accels) / len(step_accels) if step_accels else 0.0
    else:
        mean_acceleration_px_per_sec2 = 0.0

    # Direction change counting: filter out jitter/noise below min_movement_px
    sig_vectors = [
        (vx, vy) for (vx, vy) in motion_vectors
        if math.hypot(vx, vy) >= cfg.min_movement_px
    ]

    direction_change_count = 0
    threshold_rad = math.radians(cfg.direction_change_threshold_deg)

    for i in range(len(sig_vectors) - 1):
        v1 = sig_vectors[i]
        v2 = sig_vectors[i + 1]
        norm1 = math.hypot(v1[0], v1[1])
        norm2 = math.hypot(v2[0], v2[1])

        if norm1 > 0.0 and norm2 > 0.0:
            dot = v1[0] * v2[0] + v1[1] * v2[1]
            cos_theta = max(-1.0, min(1.0, dot / (norm1 * norm2)))
            angle_rad = math.acos(cos_theta)
            if angle_rad >= threshold_rad:
                direction_change_count += 1

    return TemporalFeatures(
        track_id=track_id,
        target_class=target_cls,
        duration_seconds=round(duration_seconds, 3),
        observation_count=observation_count,
        total_frames=total_frames,
        visibility_ratio=round(visibility_ratio, 3),
        detection_gap_count=detection_gap_count,
        max_consecutive_gap=max_consecutive_gap,
        start_centroid=start_centroid,
        end_centroid=end_centroid,
        total_displacement_px=round(total_displacement_px, 2),
        total_path_length_px=round(total_path_length_px, 2),
        mean_speed_px_per_sec=round(mean_speed_px_per_sec, 2),
        max_speed_px_per_sec=round(max_speed_px_per_sec, 2),
        mean_acceleration_px_per_sec2=round(mean_acceleration_px_per_sec2, 2),
        direction_change_count=direction_change_count,
        min_bbox_area_px2=round(min_bbox_area_px2, 1),
        max_bbox_area_px2=round(max_bbox_area_px2, 1),
        bbox_growth_ratio=round(bbox_growth_ratio, 2),
        mean_confidence=round(mean_confidence, 3),
    )


def format_track_diagnostics(features: TemporalFeatures) -> str:
    """Format extracted temporal features into a standardized human-readable summary."""
    start_str = f"({features.start_centroid[0]:.1f}, {features.start_centroid[1]:.1f})" if features.start_centroid else "None"
    end_str = f"({features.end_centroid[0]:.1f}, {features.end_centroid[1]:.1f})" if features.end_centroid else "None"
    gap_unit = "frame" if features.max_consecutive_gap == 1 else "frames"

    lines = [
        f"Track {features.track_id} ({features.target_class.value.upper()})",
        "-" * 40,
        f"Duration:              {features.duration_seconds:.2f} s",
        f"Observations:          {features.observation_count}",
        f"Lifespan frames:       {features.total_frames}",
        f"Visibility ratio:      {features.visibility_ratio:.2f}",
        f"Detection gaps:        {features.detection_gap_count}",
        f"Max gap:               {features.max_consecutive_gap} {gap_unit}",
        "",
        f"Start centroid:        {start_str}",
        f"End centroid:          {end_str}",
        f"Total displacement:    {features.total_displacement_px:.1f} px",
        f"Total path length:     {features.total_path_length_px:.1f} px",
        "",
        f"Mean speed:            {features.mean_speed_px_per_sec:.1f} px/s",
        f"Max speed:             {features.max_speed_px_per_sec:.1f} px/s",
        f"Mean acceleration:     {features.mean_acceleration_px_per_sec2:.1f} px/s²",
        "",
        f"Direction changes:     {features.direction_change_count}",
        "",
        "BBox area:",
        f"  min:                 {features.min_bbox_area_px2:.0f} px²",
        f"  max:                 {features.max_bbox_area_px2:.0f} px²",
        f"  growth ratio:        {features.bbox_growth_ratio:.2f}",
        "",
        f"Mean confidence:       {features.mean_confidence:.2f}",
        "-" * 40,
    ]
    return "\n".join(lines)
