"""Detector evaluation package."""

from src.evaluation.detector_video import (
    ClassDetectionMetrics,
    DetectorVideoEvaluationReport,
    FrameDetectionSummary,
    OverlapConfusionMetrics,
    VideoMetadata,
    aggregate_detector_metrics,
    evaluate_detector_on_video,
    format_detector_evaluation_report,
)

__all__ = [
    "VideoMetadata",
    "FrameDetectionSummary",
    "ClassDetectionMetrics",
    "OverlapConfusionMetrics",
    "DetectorVideoEvaluationReport",
    "aggregate_detector_metrics",
    "evaluate_detector_on_video",
    "format_detector_evaluation_report",
]
