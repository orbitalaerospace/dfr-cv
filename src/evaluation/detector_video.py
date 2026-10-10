"""Detector-only video evaluation module.

Evaluates an object detection model frame-by-frame on a video stream without
ByteTrack, temporal analysis, or behavior classification, recording class
frequencies, confidences, bounding boxes, and class overlap statistics.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple
import cv2
from pydantic import BaseModel, ConfigDict, Field

from src.detectors.ultralytics_adapter import UltralyticsDetectorAdapter
from src.domain.enums import TargetClass
from src.schemas.detection import Detection


class VideoMetadata(BaseModel):
    """Metadata describing the evaluated video source."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_path: str = Field(..., description="Path to evaluated video file")
    width: int = Field(..., ge=0, description="Video width in pixels")
    height: int = Field(..., ge=0, description="Video height in pixels")
    fps: float = Field(..., ge=0.0, description="Native video frame rate")
    total_frames: int = Field(..., ge=0, description="Total frame count in video")


class FrameDetectionSummary(BaseModel):
    """Per-frame detection breakdown recording counts, confidences, and bounding boxes."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    frame_index: int = Field(..., ge=0, description="Zero-indexed frame counter")
    timestamp: datetime = Field(..., description="Frame observation timestamp")
    num_swimmers: int = Field(..., ge=0, description="Count of SWIMMER detections in frame")
    num_watercraft: int = Field(..., ge=0, description="Count of WATERCRAFT detections in frame")
    swimmer_confidences: List[float] = Field(default_factory=list, description="Confidence values for SWIMMER detections")
    watercraft_confidences: List[float] = Field(default_factory=list, description="Confidence values for WATERCRAFT detections")
    swimmer_bboxes: List[Tuple[float, float, float, float]] = Field(
        default_factory=list,
        description="Bounding boxes (x1, y1, x2, y2) for SWIMMER detections",
    )
    watercraft_bboxes: List[Tuple[float, float, float, float]] = Field(
        default_factory=list,
        description="Bounding boxes (x1, y1, x2, y2) for WATERCRAFT detections",
    )
    all_classes: List[str] = Field(default_factory=list, description="List of all detected class names in frame")
    has_swimmer: bool = Field(..., description="Whether frame contains at least 1 SWIMMER detection")
    has_watercraft: bool = Field(..., description="Whether frame contains at least 1 WATERCRAFT detection")
    has_both: bool = Field(..., description="Whether frame contains both SWIMMER and WATERCRAFT detections")
    has_neither: bool = Field(..., description="Whether frame contains neither SWIMMER nor WATERCRAFT detections")


class ClassDetectionMetrics(BaseModel):
    """Aggregated detection metrics for a single target class across the video."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    target_class: TargetClass = Field(..., description="Target class evaluated")
    frames_detected: int = Field(..., ge=0, description="Number of frames with >=1 detection of this class")
    detection_rate: float = Field(..., ge=0.0, le=1.0, description="Fraction of total frames containing this class")
    total_detections: int = Field(..., ge=0, description="Total detection instances emitted for this class")
    mean_confidence: float = Field(..., ge=0.0, le=1.0, description="Mean confidence score for detections of this class")
    min_confidence: float = Field(..., ge=0.0, le=1.0, description="Minimum confidence score observed")
    max_confidence: float = Field(..., ge=0.0, le=1.0, description="Maximum confidence score observed")
    frames_not_detected: int = Field(default=0, ge=0, description="Number of frames with 0 detections of this class")
    avg_detections_per_frame: float = Field(default=0.0, ge=0.0, description="Total detections divided by total frames")


class OverlapConfusionMetrics(BaseModel):
    """Co-occurrence and confusion metrics between SWIMMER and WATERCRAFT classes."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    frames_with_both: int = Field(..., ge=0, description="Frames containing both SWIMMER and WATERCRAFT")
    frames_with_neither: int = Field(..., ge=0, description="Frames containing neither SWIMMER nor WATERCRAFT")
    frames_with_swimmer_only: int = Field(..., ge=0, description="Frames containing SWIMMER but no WATERCRAFT")
    frames_with_watercraft_only: int = Field(..., ge=0, description="Frames containing WATERCRAFT but no SWIMMER")


class DetectorVideoEvaluationReport(BaseModel):
    """Complete structured evaluation report for a detector evaluated on a video."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    video_metadata: VideoMetadata
    detector_name: str
    conf_threshold: float
    classes_filter: Optional[List[int]] = Field(default=None, description="Native class IDs filtered during inference")
    swimmer_metrics: ClassDetectionMetrics
    watercraft_metrics: ClassDetectionMetrics
    overlap_metrics: OverlapConfusionMetrics
    per_class_counts: Dict[str, int]
    per_frame_summaries: List[FrameDetectionSummary]


def aggregate_detector_metrics(
    frame_records: Sequence[Tuple[int, datetime, Sequence[Detection]]],
    video_metadata: VideoMetadata,
    detector_name: str,
    conf_threshold: float,
    classes_filter: Optional[List[int]] = None,
) -> DetectorVideoEvaluationReport:
    """Pure aggregation function computing detector performance metrics from frame detections.

    Args:
        frame_records: Sequence of (frame_index, timestamp, detections) tuples.
        video_metadata: Metadata describing video dimensions, fps, and frame count.
        detector_name: Label identifier for the evaluated detector.
        conf_threshold: Confidence threshold used during evaluation.
        classes_filter: Optional list of native class IDs filtered during inference.

    Returns:
        Immutable DetectorVideoEvaluationReport containing aggregate statistics and frame summaries.
    """
    total_frames = video_metadata.total_frames or len(frame_records)

    per_frame_summaries: List[FrameDetectionSummary] = []
    swimmer_confidences: List[float] = []
    watercraft_confidences: List[float] = []
    per_class_counts: Dict[str, int] = {}

    frames_with_swimmer = 0
    frames_with_watercraft = 0
    frames_with_both = 0
    frames_with_neither = 0
    frames_with_swimmer_only = 0
    frames_with_watercraft_only = 0

    for frame_index, timestamp, detections in frame_records:
        swimmers: List[Detection] = []
        watercraft: List[Detection] = []
        frame_classes: List[str] = []

        for det in detections:
            cls_name = det.target_class.value
            per_class_counts[cls_name] = per_class_counts.get(cls_name, 0) + 1
            frame_classes.append(cls_name)

            if det.target_class == TargetClass.SWIMMER:
                swimmers.append(det)
                swimmer_confidences.append(float(det.confidence))
            elif det.target_class == TargetClass.WATERCRAFT:
                watercraft.append(det)
                watercraft_confidences.append(float(det.confidence))

        has_s = len(swimmers) > 0
        has_w = len(watercraft) > 0
        has_b = has_s and has_w
        has_n = not has_s and not has_w

        if has_s:
            frames_with_swimmer += 1
        if has_w:
            frames_with_watercraft += 1
        if has_b:
            frames_with_both += 1
        if has_n:
            frames_with_neither += 1
        if has_s and not has_w:
            frames_with_swimmer_only += 1
        if has_w and not has_s:
            frames_with_watercraft_only += 1

        summary = FrameDetectionSummary(
            frame_index=frame_index,
            timestamp=timestamp,
            num_swimmers=len(swimmers),
            num_watercraft=len(watercraft),
            swimmer_confidences=[round(float(d.confidence), 4) for d in swimmers],
            watercraft_confidences=[round(float(d.confidence), 4) for d in watercraft],
            swimmer_bboxes=[d.bbox.as_xyxy() for d in swimmers],
            watercraft_bboxes=[d.bbox.as_xyxy() for d in watercraft],
            all_classes=frame_classes,
            has_swimmer=has_s,
            has_watercraft=has_w,
            has_both=has_b,
            has_neither=has_n,
        )
        per_frame_summaries.append(summary)

    # Calculate SWIMMER metrics
    swimmer_metrics = ClassDetectionMetrics(
        target_class=TargetClass.SWIMMER,
        frames_detected=frames_with_swimmer,
        detection_rate=round(frames_with_swimmer / max(1, total_frames), 4),
        total_detections=len(swimmer_confidences),
        mean_confidence=round(sum(swimmer_confidences) / len(swimmer_confidences), 4) if swimmer_confidences else 0.0,
        min_confidence=round(min(swimmer_confidences), 4) if swimmer_confidences else 0.0,
        max_confidence=round(max(swimmer_confidences), 4) if swimmer_confidences else 0.0,
        frames_not_detected=total_frames - frames_with_swimmer,
        avg_detections_per_frame=round(len(swimmer_confidences) / max(1, total_frames), 4),
    )

    # Calculate WATERCRAFT metrics
    watercraft_metrics = ClassDetectionMetrics(
        target_class=TargetClass.WATERCRAFT,
        frames_detected=frames_with_watercraft,
        detection_rate=round(frames_with_watercraft / max(1, total_frames), 4),
        total_detections=len(watercraft_confidences),
        mean_confidence=round(sum(watercraft_confidences) / len(watercraft_confidences), 4) if watercraft_confidences else 0.0,
        min_confidence=round(min(watercraft_confidences), 4) if watercraft_confidences else 0.0,
        max_confidence=round(max(watercraft_confidences), 4) if watercraft_confidences else 0.0,
        frames_not_detected=total_frames - frames_with_watercraft,
        avg_detections_per_frame=round(len(watercraft_confidences) / max(1, total_frames), 4),
    )

    # Calculate Overlap metrics
    overlap_metrics = OverlapConfusionMetrics(
        frames_with_both=frames_with_both,
        frames_with_neither=frames_with_neither,
        frames_with_swimmer_only=frames_with_swimmer_only,
        frames_with_watercraft_only=frames_with_watercraft_only,
    )

    return DetectorVideoEvaluationReport(
        video_metadata=video_metadata,
        detector_name=detector_name,
        conf_threshold=conf_threshold,
        classes_filter=classes_filter,
        swimmer_metrics=swimmer_metrics,
        watercraft_metrics=watercraft_metrics,
        overlap_metrics=overlap_metrics,
        per_class_counts=per_class_counts,
        per_frame_summaries=per_frame_summaries,
    )


def format_detector_evaluation_report(report: DetectorVideoEvaluationReport) -> str:
    """Format evaluation metrics into the requested terminal report string."""
    vm = report.video_metadata
    sm = report.swimmer_metrics
    wm = report.watercraft_metrics
    om = report.overlap_metrics

    tot = max(1, vm.total_frames)

    lines = [
        "=" * 60,
        "DETECTOR-ONLY VIDEO EVALUATION",
        "=" * 60,
        f"Video:                 {vm.source_path}",
        f"Resolution:            {vm.width}x{vm.height}",
        f"FPS:                   {vm.fps:.1f}",
        f"Frames:                {vm.total_frames}",
    ]
    if report.classes_filter is not None:
        lines.append(f"Classes filter:        {report.classes_filter}")
    lines.extend([
        "",
        "SWIMMER",
        f"  Frames detected:     {sm.frames_detected}",
        f"  Detection rate:      {sm.detection_rate * 100:.1f}%",
        f"  Total detections:    {sm.total_detections}",
        f"  Mean confidence:     {sm.mean_confidence:.2f}",
        f"  Min confidence:      {sm.min_confidence:.2f}",
        f"  Max confidence:      {sm.max_confidence:.2f}",
        f"  Frames not detected: {sm.frames_not_detected}",
        f"  Avg dets / frame:    {sm.avg_detections_per_frame:.2f}",
        "",
        "WATERCRAFT",
        f"  Frames detected:     {wm.frames_detected}",
        f"  Detection rate:      {wm.detection_rate * 100:.1f}%",
        f"  Total detections:    {wm.total_detections}",
        f"  Mean confidence:     {wm.mean_confidence:.2f}",
        f"  Min confidence:      {wm.min_confidence:.2f}",
        f"  Max confidence:      {wm.max_confidence:.2f}",
        f"  Frames not detected: {wm.frames_not_detected}",
        f"  Avg dets / frame:    {wm.avg_detections_per_frame:.2f}",
        "",
        "OVERLAP / CONFUSION",
        f"  Frames with both:    {om.frames_with_both}",
        f"  Frames with neither: {om.frames_with_neither}",
        "=" * 60,
    ])
    return "\n".join(lines)


def evaluate_detector_on_video(
    source_path: str | Path,
    weights_path: str | Path = "models/seadronessee-yolov8n.pt",
    conf_threshold: float = 0.25,
    classes: Optional[List[int]] = None,
    imgsz: Optional[int] = None,
    device: str = "auto",
    detector_name: str = "seadronessee_yolov8n",
    max_frames: Optional[int] = None,
    export_json_path: Optional[str | Path] = None,
    print_report: bool = True,
) -> DetectorVideoEvaluationReport:
    """Evaluate detector frame-by-frame on a video file and return structured report."""
    source = Path(source_path)
    if not source.exists():
        raise FileNotFoundError(f"Source video not found: {source}")

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {source}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    detector = UltralyticsDetectorAdapter(
        weights_path=weights_path,
        conf_threshold=conf_threshold,
        classes=classes,
        imgsz=imgsz,
        device=device,
        detector_name=detector_name,
    )

    frame_records: List[Tuple[int, datetime, List[Detection]]] = []
    frame_idx = 0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if max_frames is not None and frame_idx >= max_frames:
                break

            ts = datetime.now(timezone.utc)
            dets = detector.detect(frame, frame_id=frame_idx, timestamp=ts)
            frame_records.append((frame_idx, ts, dets))
            frame_idx += 1
    finally:
        cap.release()

    video_meta = VideoMetadata(
        source_path=str(source),
        width=width,
        height=height,
        fps=fps,
        total_frames=frame_idx,
    )

    report = aggregate_detector_metrics(
        frame_records=frame_records,
        video_metadata=video_meta,
        detector_name=detector_name,
        conf_threshold=conf_threshold,
        classes_filter=classes,
    )

    if print_report:
        print(format_detector_evaluation_report(report))

    if export_json_path:
        export_p = Path(export_json_path)
        export_p.parent.mkdir(parents=True, exist_ok=True)
        with open(export_p, "w", encoding="utf-8") as f:
            json.dump(report.model_dump(mode="json"), f, indent=2)
        print(f"Detector evaluation JSON exported to: {export_p}")

    return report


def main() -> int:
    """CLI entry point for detector-only video evaluation."""
    parser = argparse.ArgumentParser(
        description="Run frame-by-frame detector-only evaluation on a video without tracking or temporal analysis.",
    )
    parser.add_argument("--source", type=str, required=True, help="Path to input video file")
    parser.add_argument("--weights", type=str, default="models/seadronessee-yolov8n.pt", help="Path to detector weights")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--classes", type=int, nargs="+", default=None, help="Filter inference to specific native class IDs (e.g. --classes 0)")
    parser.add_argument("--imgsz", type=int, default=None, help="Inference image resolution (e.g. 640, 1280)")
    parser.add_argument("--device", type=str, default="auto", help="Compute device (auto, cpu, mps, cuda)")
    parser.add_argument("--name", type=str, default="seadronessee_yolov8n", help="Detector model identifier")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional frame limit")
    parser.add_argument("--export-json", type=str, default=None, help="Path to export evaluation JSON")

    args = parser.parse_args()

    evaluate_detector_on_video(
        source_path=args.source,
        weights_path=args.weights,
        conf_threshold=args.conf,
        classes=args.classes,
        imgsz=args.imgsz,
        device=args.device,
        detector_name=args.name,
        max_frames=args.max_frames,
        export_json_path=args.export_json,
        print_report=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
