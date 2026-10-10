"""Unit tests for detector root-cause audit experiments suite."""

import pytest

from src.domain.enums import TargetClass
from src.evaluation.audit_experiments import (
    GroundTruthBox,
    _compute_consecutive_dropouts,
    compute_iou,
    compute_map_50_95,
    evaluate_ground_truth_metrics,
)
from src.schemas.detection import BoundingBox, Detection
from datetime import datetime, timezone


def test_compute_iou():
    box1 = (0.0, 0.0, 10.0, 10.0)
    box2 = (5.0, 0.0, 15.0, 10.0)
    # Intersection: 5.0 x 10.0 = 50.0
    # Union: 100.0 + 100.0 - 50.0 = 150.0 -> IoU = 50 / 150 = 1/3 ~ 0.3333
    iou = compute_iou(box1, box2)
    assert abs(iou - (1.0 / 3.0)) < 1e-4

    # Identical boxes
    assert compute_iou(box1, box1) == 1.0

    # Disjoint boxes
    box3 = (20.0, 20.0, 30.0, 30.0)
    assert compute_iou(box1, box3) == 0.0


def test_compute_consecutive_dropouts():
    # Sequence with no drops
    assert _compute_consecutive_dropouts([True, True, True]) == 0
    # Sequence with drop of 3
    assert _compute_consecutive_dropouts([True, False, False, False, True, False]) == 3
    # Sequence all drops
    assert _compute_consecutive_dropouts([False, False, False, False]) == 4


def test_evaluate_ground_truth_metrics():
    gt = [
        GroundTruthBox(frame_index=0, x1=10, y1=10, x2=50, y2=50, width=40, height=40, aspect_ratio=1.0),
        GroundTruthBox(frame_index=1, x1=20, y1=20, x2=60, y2=60, width=40, height=40, aspect_ratio=1.0),
    ]

    now = datetime.now(timezone.utc)
    # Frame 0: perfect match
    # Frame 1: missed (FN)
    # Frame 2: false alarm (FP)
    dets = [
        [Detection(bbox=BoundingBox(x1=10, y1=10, x2=50, y2=50), target_class=TargetClass.SWIMMER, confidence=0.8, frame_id=0, timestamp=now)],
        [],
        [Detection(bbox=BoundingBox(x1=100, y1=100, x2=150, y2=150), target_class=TargetClass.SWIMMER, confidence=0.6, frame_id=2, timestamp=now)],
    ]

    res = evaluate_ground_truth_metrics(dets, gt, target_class=TargetClass.SWIMMER, iou_threshold=0.5)
    assert res["total_gt"] == 2
    assert res["true_positives"] == 1
    assert res["false_negatives"] == 1
    assert res["false_positives"] == 1
    assert res["precision"] == 0.5
    assert res["recall"] == 0.5
    assert res["f1"] == 0.5
    assert res["mean_iou"] == 1.0


def test_evaluate_ground_truth_duplicate_splits():
    gt = [
        GroundTruthBox(frame_index=0, x1=10, y1=10, x2=50, y2=50, width=40, height=40, aspect_ratio=1.0),
    ]
    now = datetime.now(timezone.utc)
    # 2 detections overlapping the same single GT target
    dets = [
        [
            Detection(bbox=BoundingBox(x1=10, y1=10, x2=50, y2=50), target_class=TargetClass.SWIMMER, confidence=0.9, frame_id=0, timestamp=now),
            Detection(bbox=BoundingBox(x1=12, y1=12, x2=48, y2=48), target_class=TargetClass.SWIMMER, confidence=0.7, frame_id=0, timestamp=now),
        ]
    ]
    res = evaluate_ground_truth_metrics(dets, gt, target_class=TargetClass.SWIMMER, iou_threshold=0.5)
    assert res["true_positives"] == 1
    assert res["duplicate_splits"] == 1
    assert res["false_positives"] == 1
    assert res["recall"] == 1.0
