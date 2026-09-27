"""Pipeline execution module for single aerial image inference."""

import argparse
from pathlib import Path
import sys
import time
import cv2

from src.detectors.ultralytics_adapter import UltralyticsDetectorAdapter
from src.visualization.annotator import draw_detections


def run_image_detection(
    source_path: str | Path,
    output_path: str | Path,
    weights_path: str | Path = "models/yolo11n.pt",
    conf_threshold: float = 0.25,
    device: str = "auto",
    detector_name: str = "generic_detector_pipeline_validation",
) -> int:
    source = Path(source_path)
    output = Path(output_path)

    if not source.exists():
        print(f"Error: Source image not found: {source}", file=sys.stderr)
        return 1

    output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading detector: {weights_path} (device: {device})...")
    detector = UltralyticsDetectorAdapter(
        weights_path=weights_path,
        conf_threshold=conf_threshold,
        device=device,
        detector_name=detector_name,
    )

    image = cv2.imread(str(source))
    if image is None:
        print(f"Error: Failed to decode image file: {source}", file=sys.stderr)
        return 1

    h, w = image.shape[:2]
    print(f"Input image resolution: {w}x{h}")

    t_start = time.perf_counter()
    detections = detector.detect(image, frame_id=0)
    t_end = time.perf_counter()

    elapsed_ms = (t_end - t_start) * 1000.0
    metrics = detector.last_metrics

    print("\n" + "=" * 50)
    print(" INFERENCE RESULTS")
    print("=" * 50)
    print(f"  Device Used          : {detector.device}")
    print(f"  Total Detections     : {len(detections)}")
    print(f"  Total Elapsed Latency: {elapsed_ms:.2f} ms")
    if metrics["inference_ms"] > 0:
        print(f"  - Preprocessing      : {metrics['preprocess_ms']:.2f} ms")
        print(f"  - Model Inference    : {metrics['inference_ms']:.2f} ms")
        print(f"  - Postprocessing     : {metrics['postprocess_ms']:.2f} ms")

    if detections:
        print("\n[Detected Objects]")
        for idx, det in enumerate(detections, 1):
            b = det.bbox
            print(
                f"  {idx}. {det.target_class.value:<14} "
                f"conf={det.confidence:.3f} "
                f"box=[{b.x1:.1f}, {b.y1:.1f}, {b.x2:.1f}, {b.y2:.1f}] "
                f"area={b.area:.0f}px"
            )
    else:
        print("\n  [No objects detected above confidence threshold]")

    header = f"DFR-CV M2 | {w}x{h} | Dets: {len(detections)} | Latency: {elapsed_ms:.1f}ms | Device: {detector.device}"
    annotated = draw_detections(image, detections, header_text=header)

    cv2.imwrite(str(output), annotated)
    print(f"\nAnnotated image saved to: {output}")
    print("=" * 50)

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run spatial detection on an aerial image")
    parser.add_argument("--source", type=str, required=True, help="Path to input image")
    parser.add_argument("--output", type=str, default="runs/m2/annotated.jpg", help="Path to save output")
    parser.add_argument("--weights", type=str, default="models/yolo11n.pt", help="Path to model weights")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--device", type=str, default="auto", help="Compute device (auto, mps, cpu, cuda)")
    parser.add_argument("--name", type=str, default="generic_detector_pipeline_validation", help="Detector label")

    args = parser.parse_args()
    return run_image_detection(
        source_path=args.source,
        output_path=args.output,
        weights_path=args.weights,
        conf_threshold=args.conf,
        device=args.device,
        detector_name=args.name,
    )


if __name__ == "__main__":
    sys.exit(main())
