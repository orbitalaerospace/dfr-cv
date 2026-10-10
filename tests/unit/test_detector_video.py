"""Unit tests for detector-only video evaluation metrics aggregation."""

from datetime import datetime, timezone
import pytest

from src.domain.enums import TargetClass
from src.evaluation.detector_video import (
    ClassDetectionMetrics,
    DetectorVideoEvaluationReport,
    FrameDetectionSummary,
    OverlapConfusionMetrics,
    VideoMetadata,
    aggregate_detector_metrics,
    format_detector_evaluation_report,
)
from src.schemas.detection import BoundingBox, Detection


def _make_det(target_class: TargetClass, conf: float, frame_id: int) -> Detection:
    return Detection(
        bbox=BoundingBox(x1=10.0, y1=10.0, x2=50.0, y2=50.0),
        target_class=target_class,
        confidence=conf,
        frame_id=frame_id,
        timestamp=datetime.now(timezone.utc),
    )


def test_aggregate_detector_metrics_empty():
    meta = VideoMetadata(
        source_path="mock.mp4",
        width=1280,
        height=720,
        fps=30.0,
        total_frames=0,
    )
    report = aggregate_detector_metrics(
        frame_records=[],
        video_metadata=meta,
        detector_name="test_det",
        conf_threshold=0.25,
    )

    assert report.video_metadata.total_frames == 0
    assert report.swimmer_metrics.frames_detected == 0
    assert report.swimmer_metrics.detection_rate == 0.0
    assert report.swimmer_metrics.total_detections == 0
    assert report.swimmer_metrics.mean_confidence == 0.0
    assert report.watercraft_metrics.frames_detected == 0
    assert report.overlap_metrics.frames_with_both == 0
    assert report.overlap_metrics.frames_with_neither == 0
    assert len(report.per_frame_summaries) == 0


def test_aggregate_detector_metrics_frames_without_detections():
    meta = VideoMetadata(
        source_path="mock.mp4",
        width=1280,
        height=720,
        fps=30.0,
        total_frames=5,
    )
    ts = datetime.now(timezone.utc)
    records = [(i, ts, []) for i in range(5)]

    report = aggregate_detector_metrics(
        frame_records=records,
        video_metadata=meta,
        detector_name="test_det",
        conf_threshold=0.25,
    )

    assert len(report.per_frame_summaries) == 5
    assert report.swimmer_metrics.frames_detected == 0
    assert report.swimmer_metrics.detection_rate == 0.0
    assert report.watercraft_metrics.frames_detected == 0
    assert report.overlap_metrics.frames_with_neither == 5
    assert report.overlap_metrics.frames_with_both == 0


def test_aggregate_detector_metrics_mixed_classes():
    meta = VideoMetadata(
        source_path="mock.mp4",
        width=1280,
        height=720,
        fps=25.0,
        total_frames=4,
    )
    ts = datetime.now(timezone.utc)

    # Frame 0: Swimmer only (conf 0.80)
    # Frame 1: Watercraft only (conf 0.60)
    # Frame 2: Both Swimmer (conf 0.90) and Watercraft (conf 0.70)
    # Frame 3: Neither (empty)
    records = [
        (0, ts, [_make_det(TargetClass.SWIMMER, 0.80, 0)]),
        (1, ts, [_make_det(TargetClass.WATERCRAFT, 0.60, 1)]),
        (2, ts, [
            _make_det(TargetClass.SWIMMER, 0.90, 2),
            _make_det(TargetClass.WATERCRAFT, 0.70, 2),
        ]),
        (3, ts, []),
    ]

    report = aggregate_detector_metrics(
        frame_records=records,
        video_metadata=meta,
        detector_name="test_det",
        conf_threshold=0.25,
    )

    # Swimmer metrics: 2 frames detected (frame 0, 2) out of 4 -> 0.50
    assert report.swimmer_metrics.frames_detected == 2
    assert report.swimmer_metrics.detection_rate == 0.50
    assert report.swimmer_metrics.total_detections == 2
    assert report.swimmer_metrics.min_confidence == 0.80
    assert report.swimmer_metrics.max_confidence == 0.90
    assert report.swimmer_metrics.mean_confidence == 0.85
    assert report.swimmer_metrics.frames_not_detected == 2
    assert report.swimmer_metrics.avg_detections_per_frame == 0.50

    # Watercraft metrics: 2 frames detected (frame 1, 2) out of 4 -> 0.50
    assert report.watercraft_metrics.frames_detected == 2
    assert report.watercraft_metrics.detection_rate == 0.50
    assert report.watercraft_metrics.total_detections == 2
    assert report.watercraft_metrics.min_confidence == 0.60
    assert report.watercraft_metrics.max_confidence == 0.70
    assert report.watercraft_metrics.mean_confidence == 0.65
    assert report.watercraft_metrics.frames_not_detected == 2
    assert report.watercraft_metrics.avg_detections_per_frame == 0.50

    # Overlap / confusion
    assert report.overlap_metrics.frames_with_both == 1  # Frame 2
    assert report.overlap_metrics.frames_with_neither == 1  # Frame 3
    assert report.overlap_metrics.frames_with_swimmer_only == 1  # Frame 0
    assert report.overlap_metrics.frames_with_watercraft_only == 1  # Frame 1

    # Per-class counts
    assert report.per_class_counts[TargetClass.SWIMMER.value] == 2
    assert report.per_class_counts[TargetClass.WATERCRAFT.value] == 2

    # Per-frame summaries
    f0 = report.per_frame_summaries[0]
    assert f0.num_swimmers == 1
    assert f0.num_watercraft == 0
    assert f0.has_swimmer is True
    assert f0.has_watercraft is False
    assert f0.has_both is False
    assert f0.has_neither is False
    assert len(f0.swimmer_bboxes) == 1
    assert f0.swimmer_confidences == [0.80]

    f2 = report.per_frame_summaries[2]
    assert f2.has_both is True
    assert f2.num_swimmers == 1
    assert f2.num_watercraft == 1


def test_format_detector_evaluation_report():
    meta = VideoMetadata(
        source_path="data/samples/test.mp4",
        width=1920,
        height=1080,
        fps=30.0,
        total_frames=100,
    )
    sm = ClassDetectionMetrics(
        target_class=TargetClass.SWIMMER,
        frames_detected=50,
        detection_rate=0.50,
        total_detections=55,
        mean_confidence=0.75,
        min_confidence=0.40,
        max_confidence=0.92,
    )
    wm = ClassDetectionMetrics(
        target_class=TargetClass.WATERCRAFT,
        frames_detected=20,
        detection_rate=0.20,
        total_detections=22,
        mean_confidence=0.65,
        min_confidence=0.35,
        max_confidence=0.88,
    )
    om = OverlapConfusionMetrics(
        frames_with_both=10,
        frames_with_neither=40,
        frames_with_swimmer_only=40,
        frames_with_watercraft_only=10,
    )
    report = DetectorVideoEvaluationReport(
        video_metadata=meta,
        detector_name="test_yolo",
        conf_threshold=0.25,
        swimmer_metrics=sm,
        watercraft_metrics=wm,
        overlap_metrics=om,
        per_class_counts={"swimmer": 55, "watercraft": 22},
        per_frame_summaries=[],
    )

    text = format_detector_evaluation_report(report)
    assert "DETECTOR-ONLY VIDEO EVALUATION" in text
    assert "Video:                 data/samples/test.mp4" in text
    assert "Resolution:            1920x1080" in text
    assert "Frames detected:     50" in text
    assert "Detection rate:      50.0%" in text
    assert "Frames with both:    10" in text
    assert "Frames with neither: 40" in text


def test_aggregate_detector_metrics_with_classes_filter():
    meta = VideoMetadata(
        source_path="mock.mp4",
        width=1280,
        height=720,
        fps=25.0,
        total_frames=2,
    )
    report = aggregate_detector_metrics(
        frame_records=[],
        video_metadata=meta,
        detector_name="test_det",
        conf_threshold=0.20,
        classes_filter=[0],
    )
    assert report.classes_filter == [0]
    assert report.conf_threshold == 0.20
    text = format_detector_evaluation_report(report)
    assert "Classes filter:        [0]" in text

