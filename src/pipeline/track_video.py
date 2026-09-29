import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Dict, List, Set
import cv2

from src.analysis.behavior_classifier import TemporalDistressClassifier
from src.analysis.temporal_features import extract_temporal_features, format_track_diagnostics
from src.detectors.ultralytics_adapter import UltralyticsDetectorAdapter
from src.domain.enums import BehaviorState, TargetClass, TrackState
from src.schemas.behavior import BehaviorAssessment, BehaviorClassifierConfig
from src.schemas.temporal_features import TemporalFeatureConfig, TemporalFeatures
from src.schemas.tracking import Tracklet
from src.tracking.byte_tracker import ByteTracker
from src.visualization.annotator import draw_tracks



def run_video_tracking(
    source_path: str | Path,
    output_path: str | Path,
    weights_path: str | Path = "models/seadronessee-yolov8n.pt",
    conf_threshold: float = 0.25,
    device: str = "auto",
    detector_name: str = "seadronessee_yolov8n",
    high_threshold: float = 0.4,
    low_threshold: float = 0.1,
    match_threshold_high: float = 0.8,
    match_threshold_low: float = 0.5,
    max_lost: int = 30,
    min_hits: int = 3,
    max_frames: int | None = None,
    diagnostics: bool = True,
    export_json: str | Path | None = None,
    behavior_window: float = 2.0,
    behavior_stride: float = 0.5,
    distress_threshold: float = 0.55,
    confirm_windows: int = 3,
    recovery_windows: int = 3,
) -> int:
    """Execute sequential detection, tracking, and temporal behavior analysis on a video stream."""
    source = Path(source_path)
    output = Path(output_path)

    if not source.exists():
        print(f"Error: Source video not found: {source}", file=sys.stderr)
        return 1

    output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading detector: {weights_path} (device: {device})...")
    detector = UltralyticsDetectorAdapter(
        weights_path=weights_path,
        conf_threshold=conf_threshold,
        device=device,
        detector_name=detector_name,
    )

    print(f"Initializing ByteTracker (high_th={high_threshold}, low_th={low_threshold}, max_lost={max_lost}, min_hits={min_hits})...")
    tracker = ByteTracker(
        high_threshold=high_threshold,
        low_threshold=low_threshold,
        match_threshold_high=match_threshold_high,
        match_threshold_low=match_threshold_low,
        max_lost=max_lost,
        min_hits=min_hits,
        enforce_class_match=True,
        emit_coasting=True,
        emit_unconfirmed=True,
    )

    print(f"Initializing TemporalDistressClassifier (window={behavior_window}s, thresh={distress_threshold}, confirm={confirm_windows}w)...")
    behavior_cfg = BehaviorClassifierConfig(
        window_seconds=behavior_window,
        stride_seconds=behavior_stride,
        distress_threshold=distress_threshold,
        confirm_windows=confirm_windows,
        recovery_windows=recovery_windows,
    )
    behavior_classifier = TemporalDistressClassifier(config=behavior_cfg)


    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        print(f"Error: Could not open video file: {source}", file=sys.stderr)
        return 1

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Input Video: {source.name} | {width}x{height} @ {fps:.1f} FPS | Total frames: {total_frames}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output), fourcc, fps, (width, height))

    frame_id = 0
    total_detections_count = 0
    all_seen_track_ids: Set[int] = set()
    latest_tracklets: Dict[int, Tracklet] = {}
    latest_behaviors: Dict[int, BehaviorAssessment] = {}
    track_observations: Dict[int, int] = {}
    track_classes: Dict[int, TargetClass] = {}
    track_states_history: Dict[int, List[TrackState]] = {}
    frames_with_swimmers = 0
    gaps_bridged = 0

    det_latencies_ms: List[float] = []
    track_latencies_ms: List[float] = []
    frame_latencies_ms: List[float] = []

    print("\nProcessing sequential video stream (Capture -> Detect -> Track -> Behavior -> Render)...")
    t_pipeline_start = time.perf_counter()

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if max_frames and frame_id >= max_frames:
                break

            now = datetime.now(timezone.utc)
            t_frame_start = time.perf_counter()

            # 1. Spatial Detection
            t_det_0 = time.perf_counter()
            detections = detector.detect(frame, frame_id=frame_id, timestamp=now)
            t_det_1 = time.perf_counter()
            det_ms = (t_det_1 - t_det_0) * 1000.0
            det_latencies_ms.append(det_ms)
            total_detections_count += len(detections)

            # 2. Multi-Object Tracking
            t_trk_0 = time.perf_counter()
            tracklets = tracker.update(detections, frame_id=frame_id, timestamp=now)
            t_trk_1 = time.perf_counter()
            trk_ms = (t_trk_1 - t_trk_0) * 1000.0
            track_latencies_ms.append(trk_ms)

            # 3. Temporal Behavior Classification (Milestone 5)
            frame_behaviors: Dict[int, BehaviorAssessment] = {}
            for trk in tracklets:
                assessment = behavior_classifier.update(trk, default_fps=fps, current_timestamp=now)
                frame_behaviors[trk.track_id] = assessment
                latest_behaviors[trk.track_id] = assessment

            t_frame_end = time.perf_counter()
            total_frame_ms = (t_frame_end - t_frame_start) * 1000.0
            frame_latencies_ms.append(total_frame_ms)

            # Metrics aggregation
            has_swimmer = False
            for trk in tracklets:
                tid = trk.track_id
                all_seen_track_ids.add(tid)
                latest_tracklets[tid] = trk
                track_observations[tid] = track_observations.get(tid, 0) + 1
                track_classes[tid] = trk.current_detection.target_class

                if tid not in track_states_history:
                    track_states_history[tid] = []
                prev_state = track_states_history[tid][-1] if track_states_history[tid] else None
                track_states_history[tid].append(trk.state)

                # Check if a gap was bridged (transitioned from COASTING back to TRACKED)
                if prev_state == TrackState.COASTING and trk.state == TrackState.TRACKED:
                    gaps_bridged += 1

                if trk.current_detection.target_class == TargetClass.SWIMMER:
                    has_swimmer = True

            if has_swimmer:
                frames_with_swimmers += 1

            # 4. Visualization
            curr_fps = 1000.0 / total_frame_ms if total_frame_ms > 0 else 0.0
            header = (
                f"Frame {frame_id:04d} | Tracks: {len(tracklets)} (All: {len(all_seen_track_ids)}) | "
                f"Det: {det_ms:.1f}ms | Trk: {trk_ms:.2f}ms | {curr_fps:.1f} FPS"
            )
            annotated = draw_tracks(frame, tracklets, behaviors=frame_behaviors, header_text=header)
            out.write(annotated)


            frame_id += 1

    finally:
        cap.release()
        out.release()

    t_pipeline_end = time.perf_counter()
    total_time_sec = t_pipeline_end - t_pipeline_start
    effective_fps = frame_id / total_time_sec if total_time_sec > 0 else 0.0
    avg_det_ms = sum(det_latencies_ms) / len(det_latencies_ms) if det_latencies_ms else 0.0
    avg_trk_ms = sum(track_latencies_ms) / len(track_latencies_ms) if track_latencies_ms else 0.0
    avg_total_ms = sum(frame_latencies_ms) / len(frame_latencies_ms) if frame_latencies_ms else 0.0

    # Class breakdown of tracked targets
    swimmer_tracks = [tid for tid, cls in track_classes.items() if cls == TargetClass.SWIMMER]
    watercraft_tracks = [tid for tid, cls in track_classes.items() if cls == TargetClass.WATERCRAFT]

    print("\n" + "=" * 60)
    print(" TRACKING BENCHMARK REPORT (MILESTONE 4)")
    print("=" * 60)
    print(f"  Frames Processed         : {frame_id}")
    print(f"  Input Video Native FPS   : {fps:.1f} FPS")
    print(f"  Total Wall Clock Time    : {total_time_sec:.2f} s")
    print(f"  Effective Pipeline FPS   : {effective_fps:.2f} FPS")
    print(f"  Mean Frame Latency       : {avg_total_ms:.2f} ms")
    print(f"    - Detection Latency    : {avg_det_ms:.2f} ms")
    print(f"    - ByteTrack Latency    : {avg_trk_ms:.2f} ms ({avg_trk_ms/max(1e-3, avg_total_ms)*100:.1f}%)")
    print(f"  Compute Device           : {detector.device} (CPU for ByteTrack)")
    print(f"  Total Detections Emitted : {total_detections_count}")
    print(f"  Total Unique Tracks      : {len(all_seen_track_ids)}")
    print(f"    - Swimmer Tracks       : {len(swimmer_tracks)} (IDs: {swimmer_tracks[:10]}{'...' if len(swimmer_tracks)>10 else ''})")
    print(f"    - Watercraft Tracks    : {len(watercraft_tracks)} (IDs: {watercraft_tracks[:10]}{'...' if len(watercraft_tracks)>10 else ''})")
    print(f"  Frames with Swimmers     : {frames_with_swimmers} ({frames_with_swimmers/max(1, frame_id)*100:.1f}%)")
    print(f"  Detection Gaps Bridged   : {gaps_bridged}")

    if track_observations:
        obs_counts = list(track_observations.values())
        print(f"  Observation Length (Frames): min={min(obs_counts)}, max={max(obs_counts)}, mean={sum(obs_counts)/len(obs_counts):.1f}")
    print(f"\nAnnotated tracking video saved to: {output}")
    print("=" * 60)

    # Milestone 4.5: Track Diagnostics & Temporal Feature Extraction
    temporal_features_map: Dict[int, TemporalFeatures] = {}
    config = TemporalFeatureConfig(default_fps=fps)
    for tid, trk in sorted(latest_tracklets.items()):
        feat = extract_temporal_features(trk, config=config)
        temporal_features_map[tid] = feat

    if diagnostics and temporal_features_map:
        print("\n" + "=" * 60)
        print(" TRACK DIAGNOSTICS & TEMPORAL FEATURES (MILESTONE 4.5)")
        print("=" * 60)
        for tid, feat in sorted(temporal_features_map.items()):
            print(format_track_diagnostics(feat))
            print()

    # Milestone 5: Temporal Behavior Summary
    distress_confirmed = [tid for tid, b in latest_behaviors.items() if b.state == BehaviorState.DISTRESS_CONFIRMED]
    distress_candidate = [tid for tid, b in latest_behaviors.items() if b.state == BehaviorState.DISTRESS_CANDIDATE]
    normal_swimmers = [
        tid for tid, b in latest_behaviors.items()
        if b.state == BehaviorState.NORMAL and b.target_class in (TargetClass.SWIMMER, TargetClass.PERSON, TargetClass.PERSON_SURFACE)
    ]
    unknown_swimmers = [
        tid for tid, b in latest_behaviors.items()
        if b.state == BehaviorState.UNKNOWN and b.target_class in (TargetClass.SWIMMER, TargetClass.PERSON, TargetClass.PERSON_SURFACE)
    ]

    print("\n" + "=" * 60)
    print(" TEMPORAL BEHAVIOR ANALYSIS REPORT (MILESTONE 5)")
    print("=" * 60)
    print(f"  Swimmer Tracks Evaluated : {len(swimmer_tracks)}")
    print(f"    - Distress Confirmed   : {len(distress_confirmed)} (IDs: {distress_confirmed})")
    print(f"    - Distress Candidate   : {len(distress_candidate)} (IDs: {distress_candidate})")
    print(f"    - Normal Swimming      : {len(normal_swimmers)} (IDs: {normal_swimmers})")
    print(f"    - Unknown / Sparse     : {len(unknown_swimmers)} (IDs: {unknown_swimmers})")
    print("\n  Track Behavior Details:")
    for tid in sorted(latest_behaviors.keys()):
        b = latest_behaviors[tid]
        if b.target_class in (TargetClass.SWIMMER, TargetClass.PERSON, TargetClass.PERSON_SURFACE):
            print(f"    Track #{tid:02d}: {b.state.value.upper():<18} | Distress Score: {b.distress_score:.2f} | Conf: {b.confidence:.2f}")
            print(f"      Reason: {b.explanation}")
    print("=" * 60)

    if export_json and (temporal_features_map or latest_behaviors):
        export_p = Path(export_json)
        export_p.parent.mkdir(parents=True, exist_ok=True)
        dump_data = {
            str(tid): {
                "track_id": tid,
                "target_class": track_classes.get(tid, TargetClass.UNKNOWN).value,
                "temporal_features": temporal_features_map[tid].model_dump(mode="json") if tid in temporal_features_map else None,
                "behavior_assessment": latest_behaviors[tid].model_dump(mode="json") if tid in latest_behaviors else None,
            }
            for tid in sorted(all_seen_track_ids)
        }
        with open(export_p, "w", encoding="utf-8") as f:
            json.dump(dump_data, f, indent=2)
        print(f"Track diagnostics & behavior JSON exported to: {export_p}")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run sequential spatial detection, ByteTrack tracking, and temporal behavior analysis on an aerial video")
    parser.add_argument("--source", type=str, required=True, help="Path to input video file")
    parser.add_argument("--output", type=str, default="runs/m5/tracking.mp4", help="Path to save output video")
    parser.add_argument("--weights", type=str, default="models/seadronessee-yolov8n.pt", help="Path to model weights")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--device", type=str, default="auto", help="Compute device (auto, mps, cpu, cuda)")
    parser.add_argument("--name", type=str, default="seadronessee_yolov8n", help="Detector label")
    parser.add_argument("--high-thresh", type=float, default=0.4, help="ByteTrack high confidence threshold")
    parser.add_argument("--low-thresh", type=float, default=0.1, help="ByteTrack low confidence threshold")
    parser.add_argument("--match-thresh", type=float, default=0.8, help="ByteTrack match distance threshold (1 - IoU)")
    parser.add_argument("--max-lost", type=int, default=30, help="Max coasting frames before dropping track")
    parser.add_argument("--min-hits", type=int, default=3, help="Min hits before track confirmation")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional frame limit")
    parser.add_argument("--diagnostics", action="store_true", default=True, help="Print track diagnostics summary")
    parser.add_argument("--no-diagnostics", dest="diagnostics", action="store_false", help="Disable track diagnostics")
    parser.add_argument("--export-json", type=str, default=None, help="Path to save track diagnostics & behavior JSON")
    parser.add_argument("--behavior-window", type=float, default=2.0, help="Rolling temporal window duration (seconds)")
    parser.add_argument("--behavior-stride", type=float, default=0.5, help="Rolling temporal window stride (seconds)")
    parser.add_argument("--distress-thresh", type=float, default=0.55, help="Distress behavior score threshold")
    parser.add_argument("--confirm-windows", type=int, default=3, help="Consecutive windows to confirm distress")
    parser.add_argument("--recovery-windows", type=int, default=3, help="Consecutive windows to de-escalate distress")

    args = parser.parse_args()
    return run_video_tracking(
        source_path=args.source,
        output_path=args.output,
        weights_path=args.weights,
        conf_threshold=args.conf,
        device=args.device,
        detector_name=args.name,
        high_threshold=args.high_thresh,
        low_threshold=args.low_thresh,
        match_threshold_high=args.match_thresh,
        match_threshold_low=0.5,
        max_lost=args.max_lost,
        min_hits=args.min_hits,
        max_frames=args.max_frames,
        diagnostics=args.diagnostics,
        export_json=args.export_json,
        behavior_window=args.behavior_window,
        behavior_stride=args.behavior_stride,
        distress_threshold=args.distress_thresh,
        confirm_windows=args.confirm_windows,
        recovery_windows=args.recovery_windows,
    )


if __name__ == "__main__":
    sys.exit(main())


