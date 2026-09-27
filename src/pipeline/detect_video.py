"""Pipeline execution module for sequential aerial video stream inference."""

import argparse
from pathlib import Path
import sys
import time
from typing import Dict, List
import cv2

from src.detectors.ultralytics_adapter import UltralyticsDetectorAdapter
from src.schemas.detection import Detection
from src.visualization.annotator import draw_detections


def run_video_detection(
    source_path: str | Path,
    output_path: str | Path,
    weights_path: str | Path = "models/yolo11n.pt",
    conf_threshold: float = 0.25,
    device: str = "auto",
    detector_name: str = "generic_detector_pipeline_validation",
    max_frames: int | None = None,
) -> int:
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

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        print(f"Error: Could not open video file: {source}", file=sys.stderr)
        return 1

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Input Video: {source.name} | {width}x{height} @ {fps:.1f} FPS | Total frames: {total_frames}")

    # Use mp4v codec for cross-platform compatibility
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output), fourcc, fps, (width, height))

    frame_id = 0
    frames_with_detections = 0
    total_detections_count = 0
    class_counts: Dict[str, int] = {}
    confidences: List[float] = []
    latencies_ms: List[float] = []

    print("\nProcessing video stream...")
    t_pipeline_start = time.perf_counter()

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if max_frames and frame_id >= max_frames:
                break

            t0 = time.perf_counter()
            detections = detector.detect(frame, frame_id=frame_id)
            t1 = time.perf_counter()

            frame_latency_ms = (t1 - t0) * 1000.0
            latencies_ms.append(frame_latency_ms)

            if detections:
                frames_with_detections += 1
                total_detections_count += len(detections)
                for d in detections:
                    c_name = d.target_class.value
                    class_counts[c_name] = class_counts.get(c_name, 0) + 1
                    confidences.append(d.confidence)

            # Render overlay
            curr_fps = 1000.0 / frame_latency_ms if frame_latency_ms > 0 else 0.0
            header = (
                f"Frame {frame_id:04d} | Dets: {len(detections)} | "
                f"Lat: {frame_latency_ms:.1f}ms ({curr_fps:.1f} FPS) | {detector.device}"
            )
            annotated = draw_detections(frame, detections, header_text=header)
            out.write(annotated)

            frame_id += 1

    finally:
        cap.release()
        out.release()

    t_pipeline_end = time.perf_counter()
    total_time_sec = t_pipeline_end - t_pipeline_start
    effective_fps = frame_id / total_time_sec if total_time_sec > 0 else 0.0
    avg_latency = sum(latencies_ms) / len(latencies_ms) if latencies_ms else 0.0

    print("\n" + "=" * 55)
    print(" VIDEO INFERENCE BENCHMARK REPORT")
    print("=" * 55)
    print(f"  Frames Processed         : {frame_id}")
    print(f"  Total Wall Time          : {total_time_sec:.2f} s")
    print(f"  Effective Pipeline FPS   : {effective_fps:.2f} FPS")
    print(f"  Mean Inference Latency   : {avg_latency:.2f} ms")
    print(f"  Compute Device           : {detector.device}")
    print(f"  Frames with Detections   : {frames_with_detections} ({frames_with_detections/max(1, frame_id)*100:.1f}%)")
    print(f"  Total Detections Emitted : {total_detections_count}")

    if confidences:
        print(f"  Observed Confidence Range: {min(confidences):.3f} - {max(confidences):.3f} (mean: {sum(confidences)/len(confidences):.3f})")
    print(f"  Class Breakdown          : {class_counts}")
    print(f"\nAnnotated video saved to: {output}")
    print("=" * 55)

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run spatial detection on an aerial video stream")
    parser.add_argument("--source", type=str, required=True, help="Path to input video file")
    parser.add_argument("--output", type=str, default="runs/m2/annotated.mp4", help="Path to save output video")
    parser.add_argument("--weights", type=str, default="models/yolo11n.pt", help="Path to model weights")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--device", type=str, default="auto", help="Compute device (auto, mps, cpu, cuda)")
    parser.add_argument("--name", type=str, default="generic_detector_pipeline_validation", help="Detector label")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional frame limit")

    args = parser.parse_args()
    return run_video_detection(
        source_path=args.source,
        output_path=args.output,
        weights_path=args.weights,
        conf_threshold=args.conf,
        device=args.device,
        detector_name=args.name,
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    sys.exit(main())
