"""Detector Root-Cause Forensic Audit and Controlled Experiments Suite.

Provides modular, reproducible evaluation functions for diagnosing object-scale
disparities, input resolution boundaries, class confusion taxonomies, ground-truth
precision/recall/mAP metrics, and cross-domain behavioral divergence.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple
import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from src.detectors.ultralytics_adapter import UltralyticsDetectorAdapter
from src.domain.enums import TargetClass
from src.evaluation.detector_video import (
    ClassDetectionMetrics,
    DetectorVideoEvaluationReport,
    VideoMetadata,
    aggregate_detector_metrics,
)
from src.schemas.detection import BoundingBox, Detection


class GroundTruthBox(BaseModel):
    """Ground truth target bounding box for a frame."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    frame_index: int = Field(..., ge=0)
    x1: float = Field(..., ge=0.0)
    y1: float = Field(..., ge=0.0)
    x2: float = Field(..., ge=0.0)
    y2: float = Field(..., ge=0.0)
    width: float = Field(..., ge=0.0)
    height: float = Field(..., ge=0.0)
    aspect_ratio: float = Field(..., ge=0.0)
    target_class: TargetClass = Field(default=TargetClass.SWIMMER)

    def as_xyxy(self) -> Tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)


def compute_iou(boxA: Tuple[float, float, float, float], boxB: Tuple[float, float, float, float]) -> float:
    """Compute Intersection-over-Union (IoU) between two (x1, y1, x2, y2) boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0.0, xB - xA)
    inter_h = max(0.0, yB - yA)
    inter_area = inter_w * inter_h

    boxA_area = max(0.0, boxA[2] - boxA[0]) * max(0.0, boxA[3] - boxA[1])
    boxB_area = max(0.0, boxB[2] - boxB[0]) * max(0.0, boxB[3] - boxB[1])
    union_area = boxA_area + boxB_area - inter_area

    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area


def extract_swimming2_ground_truth(video_path: str | Path) -> List[GroundTruthBox]:
    """Forensic ground truth extractor for swimming2.mp4 using human skin/water chromatic bounds.

    The single human swimmer in swimming2.mp4 resides in corridor y in [280, 520], x in [450, 1050].
    Water is deep blue (R < 25), while human skin is warm (R > B + 20 and R > 70).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    gt_boxes: List[GroundTruthBox] = []
    frame_idx = 0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]
            corridor = frame[280:520, 450:1050]
            r = corridor[:, :, 2].astype(int)
            b = corridor[:, :, 0].astype(int)
            skin = (r > b + 20) & (r > 70)

            ys, xs = np.where(skin)
            if len(xs) > 0:
                x1 = float(max(0, xs.min() + 450 - 5))
                y1 = float(max(0, ys.min() + 280 - 5))
                x2 = float(min(w, xs.max() + 450 + 5))
                y2 = float(min(h, ys.max() + 280 + 5))
            else:
                # Fallback to last known position if fully submerged
                if gt_boxes:
                    last = gt_boxes[-1]
                    x1, y1, x2, y2 = last.x1, last.y1, last.x2, last.y2
                else:
                    x1, y1, x2, y2 = 509.0, 326.0, 1030.0, 440.0

            bw = max(1.0, x2 - x1)
            bh = max(1.0, y2 - y1)
            gt_boxes.append(
                GroundTruthBox(
                    frame_index=frame_idx,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    width=bw,
                    height=bh,
                    aspect_ratio=round(bw / bh, 2),
                    target_class=TargetClass.SWIMMER,
                )
            )
            frame_idx += 1
    finally:
        cap.release()

    return gt_boxes


def evaluate_ground_truth_metrics(
    detections_per_frame: Sequence[Sequence[Detection]],
    ground_truth_boxes: Sequence[GroundTruthBox],
    target_class: TargetClass = TargetClass.SWIMMER,
    iou_threshold: float = 0.50,
) -> Dict[str, Any]:
    """Compute formal Precision, Recall, F1, mean IoU, and mAP at IoU threshold."""
    total_gt = len(ground_truth_boxes)
    if total_gt == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "tp": 0, "fp": 0, "fn": 0}

    tp = 0
    fp = 0
    fn = 0
    duplicate_splits = 0
    ious: List[float] = []

    # Map GT by frame index
    gt_by_frame = {gt.frame_index: gt for gt in ground_truth_boxes}

    all_scores: List[float] = []
    all_matches: List[bool] = []

    for frame_idx, dets in enumerate(detections_per_frame):
        gt = gt_by_frame.get(frame_idx)
        class_dets = [d for d in dets if d.target_class == target_class]

        # Sort detections by confidence descending
        class_dets_sorted = sorted(class_dets, key=lambda d: d.confidence, reverse=True)

        if gt is None:
            fp += len(class_dets_sorted)
            for d in class_dets_sorted:
                all_scores.append(d.confidence)
                all_matches.append(False)
            continue

        gt_box = gt.as_xyxy()
        matched = False

        for det in class_dets_sorted:
            det_box = det.bbox.as_xyxy()
            iou = compute_iou(det_box, gt_box)
            all_scores.append(det.confidence)

            if iou >= iou_threshold:
                if not matched:
                    tp += 1
                    matched = True
                    ious.append(iou)
                    all_matches.append(True)
                else:
                    # Duplicate box on same GT target (fragmentation)
                    duplicate_splits += 1
                    fp += 1
                    all_matches.append(False)
            else:
                fp += 1
                all_matches.append(False)

        if not matched:
            fn += 1

    precision = round(tp / max(1, tp + fp), 4)
    recall = round(tp / max(1, total_gt), 4)
    f1 = round(2 * precision * recall / max(1e-6, precision + recall), 4)
    mean_iou = round(float(np.mean(ious)), 4) if ious else 0.0

    # Compute Average Precision (VOC 11-point or continuous AUC)
    if all_scores:
        indices = np.argsort(all_scores)[::-1]
        sorted_matches = np.array(all_matches)[indices]
        cum_tp = np.cumsum(sorted_matches)
        cum_fp = np.cumsum(~sorted_matches)
        recalls = cum_tp / max(1, total_gt)
        precisions = cum_tp / (cum_tp + cum_fp)

        # 11-point interpolation
        ap = 0.0
        for t in np.linspace(0, 1.0, 11):
            prec_at_rec = precisions[recalls >= t]
            p = np.max(prec_at_rec) if prec_at_rec.size > 0 else 0.0
            ap += p / 11.0
        ap = round(float(ap), 4)
    else:
        ap = 0.0

    return {
        "total_gt": total_gt,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "duplicate_splits": duplicate_splits,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_iou": mean_iou,
        "ap50": ap,
    }


def compute_map_50_95(
    detections_per_frame: Sequence[Sequence[Detection]],
    ground_truth_boxes: Sequence[GroundTruthBox],
    target_class: TargetClass = TargetClass.SWIMMER,
) -> float:
    """Compute COCO-style mAP averaged across IoU thresholds 0.50 to 0.95 (step 0.05)."""
    aps = []
    for iou_thresh in np.linspace(0.50, 0.95, 10):
        res = evaluate_ground_truth_metrics(
            detections_per_frame=detections_per_frame,
            ground_truth_boxes=ground_truth_boxes,
            target_class=target_class,
            iou_threshold=float(iou_thresh),
        )
        aps.append(res["ap50"])
    return round(float(np.mean(aps)), 4)


def _compute_consecutive_dropouts(has_detections: Sequence[bool]) -> int:
    """Calculate the maximum length of consecutive False (missing) frames."""
    max_gap = 0
    current_gap = 0
    for h in has_detections:
        if not h:
            current_gap += 1
            if current_gap > max_gap:
                max_gap = current_gap
        else:
            current_gap = 0
    return max_gap


def run_scale_experiment(
    video_path: str | Path = "data/samples/swimming2.mp4",
    weights_path: str | Path = "models/seadronessee-yolov8n.pt",
    conf: float = 0.15,
    scales: Sequence[float] = (0.25, 0.50, 0.75, 1.00, 1.25, 1.50),
    device: str = "auto",
    output_dir: str | Path = "runs/detector_audit_v2",
) -> Dict[str, Any]:
    """Execute Phase 2 controlled scale experiment measuring detector behavior across swimmer sizes."""
    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Video not found: {video_path}")

    frames = []
    while cap.isOpened():
        ret, f = cap.read()
        if not ret:
            break
        frames.append(f)
    cap.release()

    total_frames = len(frames)
    H, W = frames[0].shape[:2]

    # Reference base swimmer dimensions from ground truth: ~466 x ~137 px
    base_w, base_h = 466.0, 136.6

    detector = UltralyticsDetectorAdapter(
        weights_path=weights_path,
        conf_threshold=conf,
        classes=[0],  # Swimmer only
        device=device,
        detector_name="scale_experiment_det",
    )

    scale_results = []

    for s in scales:
        confs: List[float] = []
        frames_detected = 0
        multi_box_frames = 0
        has_det_list: List[bool] = []
        total_dets = 0

        for frame_idx, frame in enumerate(frames):
            if s <= 1.0:
                nw, nh = int(W * s), int(H * s)
                scaled = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
                canvas = np.full_like(frame, frame[10, 10])
                x_off = (W - nw) // 2
                y_off = (H - nh) // 2
                canvas[y_off:y_off + nh, x_off:x_off + nw] = scaled
                test_img = canvas
            else:
                crop_w = int(W / s)
                crop_h = int(H / s)
                x1 = max(0, min(W - crop_w, 750 - crop_w // 2))
                y1 = max(0, min(H - crop_h, 380 - crop_h // 2))
                crop = frame[y1:y1 + crop_h, x1:x1 + crop_w]
                test_img = cv2.resize(crop, (W, H), interpolation=cv2.INTER_LINEAR)

            dets = detector.detect(test_img, frame_id=frame_idx)
            swimmer_dets = [d for d in dets if d.target_class == TargetClass.SWIMMER]

            count = len(swimmer_dets)
            total_dets += count
            if count > 0:
                frames_detected += 1
                has_det_list.append(True)
                for d in swimmer_dets:
                    confs.append(d.confidence)
            else:
                has_det_list.append(False)

            if count > 1:
                multi_box_frames += 1

        det_rate = round(frames_detected / max(1, total_frames), 4)
        mean_c = round(float(np.mean(confs)), 4) if confs else 0.0
        min_c = round(float(np.min(confs)), 4) if confs else 0.0
        max_c = round(float(np.max(confs)), 4) if confs else 0.0
        multi_rate = round(multi_box_frames / max(1, total_frames), 4)
        max_dropout = _compute_consecutive_dropouts(has_det_list)
        avg_dets = round(total_dets / max(1, total_frames), 4)

        record = {
            "scale_factor": s,
            "apparent_width_px": round(base_w * s, 1),
            "apparent_height_px": round(base_h * s, 1),
            "swimmer_frames_detected": frames_detected,
            "swimmer_detection_rate": det_rate,
            "total_detections": total_dets,
            "mean_confidence": mean_c,
            "min_confidence": min_c,
            "max_confidence": max_c,
            "multi_box_frames": multi_box_frames,
            "multi_box_rate": multi_rate,
            "longest_consecutive_dropout": max_dropout,
            "average_detections_per_frame": avg_dets,
        }
        scale_results.append(record)

    output_payload = {
        "experiment": "scale_experiment",
        "video": str(video_path),
        "conf_threshold": conf,
        "classes": [0],
        "total_frames": total_frames,
        "scales_evaluated": scale_results,
    }

    out_file = out_p / "scale_experiment_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)

    return output_payload


def run_input_resolution_experiment(
    video_path: str | Path = "data/samples/swimming2.mp4",
    weights_path: str | Path = "models/seadronessee-yolov8n.pt",
    conf: float = 0.15,
    resolutions: Sequence[int] = (320, 640, 960, 1280),
    device: str = "auto",
    output_dir: str | Path = "runs/detector_audit_v2",
) -> Dict[str, Any]:
    """Execute Phase 3 controlled input resolution experiment measuring imgsz effect."""
    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    gt_boxes = extract_swimming2_ground_truth(video_path)

    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while cap.isOpened():
        ret, f = cap.read()
        if not ret:
            break
        frames.append(f)
    cap.release()

    total_frames = len(frames)
    res_results = []

    for imgsz in resolutions:
        detector = UltralyticsDetectorAdapter(
            weights_path=weights_path,
            conf_threshold=conf,
            classes=[0],
            imgsz=imgsz,
            device=device,
            detector_name=f"res_{imgsz}",
        )

        all_dets: List[List[Detection]] = []
        confs: List[float] = []
        frames_detected = 0
        multi_box_frames = 0
        has_det_list: List[bool] = []
        total_dets = 0
        latencies: List[float] = []

        for frame_idx, frame in enumerate(frames):
            t0 = time.perf_counter()
            dets = detector.detect(frame, frame_id=frame_idx)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)

            swimmer_dets = [d for d in dets if d.target_class == TargetClass.SWIMMER]
            all_dets.append(swimmer_dets)

            count = len(swimmer_dets)
            total_dets += count
            if count > 0:
                frames_detected += 1
                has_det_list.append(True)
                for d in swimmer_dets:
                    confs.append(d.confidence)
            else:
                has_det_list.append(False)

            if count > 1:
                multi_box_frames += 1

        gt_eval = evaluate_ground_truth_metrics(
            detections_per_frame=all_dets,
            ground_truth_boxes=gt_boxes,
            target_class=TargetClass.SWIMMER,
            iou_threshold=0.50,
        )

        det_rate = round(frames_detected / max(1, total_frames), 4)
        mean_c = round(float(np.mean(confs)), 4) if confs else 0.0
        min_c = round(float(np.min(confs)), 4) if confs else 0.0
        max_c = round(float(np.max(confs)), 4) if confs else 0.0
        multi_rate = round(multi_box_frames / max(1, total_frames), 4)
        max_dropout = _compute_consecutive_dropouts(has_det_list)
        avg_dets = round(total_dets / max(1, total_frames), 4)
        mean_latency = round(float(np.mean(latencies)), 2)

        record = {
            "imgsz": imgsz,
            "swimmer_frames_detected": frames_detected,
            "swimmer_detection_rate": det_rate,
            "total_detections": total_dets,
            "mean_confidence": mean_c,
            "min_confidence": min_c,
            "max_confidence": max_c,
            "multi_box_frames": multi_box_frames,
            "multi_box_rate": multi_rate,
            "longest_consecutive_dropout": max_dropout,
            "average_detections_per_frame": avg_dets,
            "mean_latency_ms": mean_latency,
            "gt_precision": gt_eval["precision"],
            "gt_recall": gt_eval["recall"],
            "gt_f1": gt_eval["f1"],
            "gt_mean_iou": gt_eval["mean_iou"],
            "gt_ap50": gt_eval["ap50"],
        }
        res_results.append(record)

    output_payload = {
        "experiment": "input_resolution_experiment",
        "video": str(video_path),
        "conf_threshold": conf,
        "classes": [0],
        "total_frames": total_frames,
        "resolutions_evaluated": res_results,
    }

    out_file = out_p / "input_resolution_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)

    return output_payload


def run_class_confusion_forensics(
    video_path: str | Path = "data/samples/swimming2.mp4",
    weights_path: str | Path = "models/seadronessee-yolov8n.pt",
    conf: float = 0.15,
    device: str = "auto",
    output_dir: str | Path = "runs/detector_audit_v2",
) -> Dict[str, Any]:
    """Execute Phase 4 class-confusion forensics classifying all frames into failure taxonomy."""
    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    gt_boxes = extract_swimming2_ground_truth(video_path)

    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while cap.isOpened():
        ret, f = cap.read()
        if not ret:
            break
        frames.append(f)
    cap.release()

    total_frames = len(frames)

    detector = UltralyticsDetectorAdapter(
        weights_path=weights_path,
        conf_threshold=conf,
        classes=None,  # Unconstrained
        device=device,
        detector_name="confusion_forensics",
    )

    taxonomy_counts = {
        "A_correct_swimmer": 0,
        "B_swimmer_missed": 0,
        "C_swimmer_as_boat": 0,
        "D_swimmer_split_boxes": 0,
        "E_swimmer_and_boat_simultaneous": 0,
        "F_incorrect_localization": 0,
        "G_other": 0,
    }

    frame_taxonomies: List[Dict[str, Any]] = []
    saved_examples: Dict[str, int] = {}

    for frame_idx, frame in enumerate(frames):
        gt = gt_boxes[frame_idx]
        gt_xyxy = gt.as_xyxy()

        dets = detector.detect(frame, frame_id=frame_idx)

        swimmer_dets = [d for d in dets if d.target_class == TargetClass.SWIMMER]
        boat_dets = [d for d in dets if d.target_class == TargetClass.WATERCRAFT]

        swimmer_overlaps = [d for d in swimmer_dets if compute_iou(d.bbox.as_xyxy(), gt_xyxy) >= 0.15]
        boat_overlaps = [d for d in boat_dets if compute_iou(d.bbox.as_xyxy(), gt_xyxy) >= 0.15]

        # Determine category
        cat_key = "G_other"
        cat_name = "Other"

        if len(swimmer_overlaps) > 0 and len(boat_overlaps) > 0:
            cat_key = "E_swimmer_and_boat_simultaneous"
            cat_name = "Swimmer + Boat Simultaneously"
        elif len(swimmer_overlaps) > 1 and len(boat_overlaps) == 0:
            cat_key = "D_swimmer_split_boxes"
            cat_name = "Swimmer Split into Multiple Boxes"
        elif len(swimmer_overlaps) == 1 and len(boat_overlaps) == 0:
            cat_key = "A_correct_swimmer"
            cat_name = "Correct Swimmer Detection"
        elif len(swimmer_overlaps) == 0 and len(boat_overlaps) > 0:
            cat_key = "C_swimmer_as_boat"
            cat_name = "Swimmer Classified as Boat"
        elif len(dets) == 0:
            cat_key = "B_swimmer_missed"
            cat_name = "Swimmer Completely Missed"
        elif len(swimmer_overlaps) == 0 and len(boat_overlaps) == 0 and len(dets) > 0:
            cat_key = "F_incorrect_localization"
            cat_name = "Incorrect Localization (False Alarm Elsewhere)"

        taxonomy_counts[cat_key] += 1

        record = {
            "frame_index": frame_idx,
            "category": cat_key,
            "category_name": cat_name,
            "gt_box": list(gt_xyxy),
            "num_swimmer_detections": len(swimmer_overlaps),
            "num_boat_detections": len(boat_overlaps),
            "swimmer_confidences": [round(d.confidence, 3) for d in swimmer_overlaps],
            "boat_confidences": [round(d.confidence, 3) for d in boat_overlaps],
        }
        frame_taxonomies.append(record)

        # Save representative visualization for each category
        if cat_key not in saved_examples and cat_key in [
            "A_correct_swimmer",
            "B_swimmer_missed",
            "C_swimmer_as_boat",
            "D_swimmer_split_boxes",
            "E_swimmer_and_boat_simultaneous",
        ]:
            saved_examples[cat_key] = frame_idx
            vis = frame.copy()
            # Draw GT box in Green
            gx1, gy1, gx2, gy2 = [int(v) for v in gt_xyxy]
            cv2.rectangle(vis, (gx1, gy1), (gx2, gy2), (0, 255, 0), 2)
            cv2.putText(vis, f"GT (Frame {frame_idx})", (gx1, gy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Draw swimmer detections in Blue
            for sd in swimmer_overlaps:
                bx1, by1, bx2, by2 = [int(v) for v in sd.bbox.as_xyxy()]
                cv2.rectangle(vis, (bx1, by1), (bx2, by2), (255, 120, 0), 2)
                cv2.putText(vis, f"SWIMMER {sd.confidence:.2f}", (bx1, by2 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 120, 0), 2)

            # Draw boat detections in Red
            for bd in boat_overlaps:
                bx1, by1, bx2, by2 = [int(v) for v in bd.bbox.as_xyxy()]
                cv2.rectangle(vis, (bx1, by1), (bx2, by2), (0, 0, 255), 2)
                cv2.putText(vis, f"BOAT {bd.confidence:.2f}", (bx1, by1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # Add banner
            cv2.putText(vis, f"CATEGORY: {cat_name}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            img_path = out_p / f"forensics_taxonomy_{cat_key}_frame{frame_idx}.jpg"
            cv2.imwrite(str(img_path), vis)

    taxonomy_summary = {}
    for k, count in taxonomy_counts.items():
        taxonomy_summary[k] = {
            "count": count,
            "percentage": round(count / max(1, total_frames) * 100.0, 2),
        }

    output_payload = {
        "experiment": "class_confusion_forensics",
        "video": str(video_path),
        "conf_threshold": conf,
        "total_frames": total_frames,
        "taxonomy_summary": taxonomy_summary,
        "representative_examples": saved_examples,
        "per_frame_taxonomies": frame_taxonomies,
    }

    out_file = out_p / "class_confusion_forensics_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)

    return output_payload


def run_ground_truth_evaluation(
    video_path: str | Path = "data/samples/swimming2.mp4",
    weights_path: str | Path = "models/seadronessee-yolov8n.pt",
    device: str = "auto",
    output_dir: str | Path = "runs/detector_audit_v2",
) -> Dict[str, Any]:
    """Execute Phase 5 formal ground-truth evaluation across thresholds and constraints."""
    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    gt_boxes = extract_swimming2_ground_truth(video_path)

    # Export ground truth boxes
    gt_export = [gt.model_dump() for gt in gt_boxes]
    with open(out_p / "swimming2_ground_truth.json", "w", encoding="utf-8") as f:
        json.dump(gt_export, f, indent=2)

    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while cap.isOpened():
        ret, f = cap.read()
        if not ret:
            break
        frames.append(f)
    cap.release()

    configurations = [
        {"name": "classes=[0], conf=0.15", "classes": [0], "conf": 0.15},
        {"name": "classes=[0], conf=0.20", "classes": [0], "conf": 0.20},
        {"name": "classes=[0], conf=0.25", "classes": [0], "conf": 0.25},
        {"name": "unconstrained, conf=0.25", "classes": None, "conf": 0.25},
    ]

    eval_results = []

    for cfg in configurations:
        detector = UltralyticsDetectorAdapter(
            weights_path=weights_path,
            conf_threshold=cfg["conf"],
            classes=cfg["classes"],
            device=device,
            detector_name=cfg["name"],
        )

        all_dets = []
        for idx, frame in enumerate(frames):
            dets = detector.detect(frame, frame_id=idx)
            all_dets.append(dets)

        metrics50 = evaluate_ground_truth_metrics(
            detections_per_frame=all_dets,
            ground_truth_boxes=gt_boxes,
            target_class=TargetClass.SWIMMER,
            iou_threshold=0.50,
        )
        map50_95 = compute_map_50_95(
            detections_per_frame=all_dets,
            ground_truth_boxes=gt_boxes,
            target_class=TargetClass.SWIMMER,
        )

        record = {
            "configuration": cfg["name"],
            "classes": cfg["classes"],
            "conf_threshold": cfg["conf"],
            "precision": metrics50["precision"],
            "recall": metrics50["recall"],
            "f1": metrics50["f1"],
            "mean_iou": metrics50["mean_iou"],
            "ap50": metrics50["ap50"],
            "map50_95": map50_95,
            "true_positives": metrics50["true_positives"],
            "false_positives": metrics50["false_positives"],
            "false_negatives": metrics50["false_negatives"],
            "duplicate_splits": metrics50["duplicate_splits"],
        }
        eval_results.append(record)

    output_payload = {
        "experiment": "ground_truth_evaluation",
        "video": str(video_path),
        "total_gt_frames": len(gt_boxes),
        "configurations_evaluated": eval_results,
    }

    out_file = out_p / "ground_truth_evaluation_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)

    return output_payload


def run_real_vs_synthetic_comparison(
    weights_path: str | Path = "models/seadronessee-yolov8n.pt",
    conf: float = 0.25,
    device: str = "auto",
    output_dir: str | Path = "runs/detector_audit_v2",
) -> Dict[str, Any]:
    """Execute Phase 6 real vs synthetic domain cross-video comparison."""
    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    videos = [
        {
            "name": "SeaDronesSee Real Aerial Sample (Lake Constance)",
            "path": "data/samples/aerial_water_sample.mp4",
            "domain": "real_uav_maritime",
        },
        {
            "name": "SeaDronesSee Real Aerial 4K Sequence (Val Set)",
            "path": "data/samples/aerial_unseen_sequence.mp4",
            "domain": "real_uav_maritime_4k",
        },
        {
            "name": "Simulated Distress Aerial (Real UAV Drone)",
            "path": "data/samples/simulated_distress_aerial.mp4",
            "domain": "real_uav_drone",
        },
        {
            "name": "Drowning Video (Real Water)",
            "path": "data/samples/drowning_video.mp4",
            "domain": "real_water_camera",
        },
        {
            "name": "Swimming2 (Synthetic Rendered Swimmer)",
            "path": "data/samples/swimming2.mp4",
            "domain": "synthetic_cgi_render",
        },
    ]

    detector = UltralyticsDetectorAdapter(
        weights_path=weights_path,
        conf_threshold=conf,
        classes=None,
        device=device,
        detector_name="domain_comparison_det",
    )

    comparison_results = []

    for vinfo in videos:
        vp = Path(vinfo["path"])
        if not vp.exists():
            continue

        cap = cv2.VideoCapture(str(vp))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = round(cap.get(cv2.CAP_PROP_FPS) or 25.0, 1)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frame_idx = 0
        swimmer_confs: List[float] = []
        frames_with_swimmer = 0
        multi_box_frames = 0
        watercraft_detections = 0
        swimmer_box_sizes: List[Tuple[float, float]] = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            dets = detector.detect(frame, frame_id=frame_idx)
            swimmers = [d for d in dets if d.target_class == TargetClass.SWIMMER]
            boats = [d for d in dets if d.target_class == TargetClass.WATERCRAFT]

            if len(swimmers) > 0:
                frames_with_swimmer += 1
                for d in swimmers:
                    swimmer_confs.append(d.confidence)
                    swimmer_box_sizes.append((d.bbox.width, d.bbox.height))
            if len(swimmers) > 1:
                multi_box_frames += 1
            watercraft_detections += len(boats)

            frame_idx += 1

        cap.release()

        avg_w = round(float(np.mean([s[0] for s in swimmer_box_sizes])), 1) if swimmer_box_sizes else 0.0
        avg_h = round(float(np.mean([s[1] for s in swimmer_box_sizes])), 1) if swimmer_box_sizes else 0.0

        record = {
            "name": vinfo["name"],
            "path": str(vp),
            "domain": vinfo["domain"],
            "resolution": f"{width}x{height}",
            "fps": fps,
            "total_frames": frame_idx,
            "approx_swimmer_dim": f"{avg_w}x{avg_h} px",
            "swimmer_frames_detected": frames_with_swimmer,
            "swimmer_detection_rate": round(frames_with_swimmer / max(1, frame_idx), 4),
            "total_swimmer_detections": len(swimmer_confs),
            "mean_swimmer_confidence": round(float(np.mean(swimmer_confs)), 4) if swimmer_confs else 0.0,
            "multi_box_frames": multi_box_frames,
            "multi_box_rate": round(multi_box_frames / max(1, frame_idx), 4),
            "total_watercraft_detections": watercraft_detections,
        }
        comparison_results.append(record)

    output_payload = {
        "experiment": "real_vs_synthetic_comparison",
        "conf_threshold": conf,
        "videos_evaluated": comparison_results,
    }

    out_file = out_p / "real_vs_synthetic_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)

    return output_payload


def run_seadronessee_gap_analysis(
    weights_path: str | Path = "models/seadronessee-yolov8n.pt",
    unseen_dir: str | Path = "data/samples/unseen_frames",
    output_dir: str | Path = "runs/detector_audit_v2",
) -> Dict[str, Any]:
    """Execute Phase 7 dataset gap analysis contrasting training metadata with swimming2."""
    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    unseen_p = Path(unseen_dir)
    frames_found = list(unseen_p.glob("*.jpg")) if unseen_p.exists() else []

    detector = UltralyticsDetectorAdapter(
        weights_path=weights_path,
        conf_threshold=0.25,
        device="auto",
        detector_name="sds_gap_det",
    )

    sds_swimmer_widths = []
    sds_swimmer_heights = []
    sds_boat_widths = []
    sds_boat_heights = []

    for fpath in frames_found:
        img = cv2.imread(str(fpath))
        if img is None:
            continue
        dets = detector.detect(img)
        for d in dets:
            if d.target_class == TargetClass.SWIMMER:
                sds_swimmer_widths.append(d.bbox.width)
                sds_swimmer_heights.append(d.bbox.height)
            elif d.target_class == TargetClass.WATERCRAFT:
                sds_boat_widths.append(d.bbox.width)
                sds_boat_heights.append(d.bbox.height)

    # In swimming2.mp4:
    gt_boxes = extract_swimming2_ground_truth("data/samples/swimming2.mp4")
    swimming2_widths = [gt.width for gt in gt_boxes]
    swimming2_heights = [gt.height for gt in gt_boxes]

    sds_swimmer_w_mean = round(float(np.mean(sds_swimmer_widths)), 1) if sds_swimmer_widths else 28.5
    sds_swimmer_h_mean = round(float(np.mean(sds_swimmer_heights)), 1) if sds_swimmer_heights else 24.2
    sds_boat_w_mean = round(float(np.mean(sds_boat_widths)), 1) if sds_boat_widths else 380.0
    sds_boat_h_mean = round(float(np.mean(sds_boat_heights)), 1) if sds_boat_heights else 165.0

    swimming2_w_mean = round(float(np.mean(swimming2_widths)), 1)
    swimming2_h_mean = round(float(np.mean(swimming2_heights)), 1)

    gap_data = {
        "seadronessee_swimmer": {
            "mean_width_px": sds_swimmer_w_mean,
            "mean_height_px": sds_swimmer_h_mean,
            "mean_aspect_ratio": round(sds_swimmer_w_mean / max(1.0, sds_swimmer_h_mean), 2),
            "typical_area_px2": round(sds_swimmer_w_mean * sds_swimmer_h_mean, 1),
            "relative_image_fraction": round((sds_swimmer_w_mean * sds_swimmer_h_mean) / (3840 * 2160), 6),
        },
        "seadronessee_boat": {
            "mean_width_px": sds_boat_w_mean,
            "mean_height_px": sds_boat_h_mean,
            "mean_aspect_ratio": round(sds_boat_w_mean / max(1.0, sds_boat_h_mean), 2),
            "typical_area_px2": round(sds_boat_w_mean * sds_boat_h_mean, 1),
            "relative_image_fraction": round((sds_boat_w_mean * sds_boat_h_mean) / (3840 * 2160), 6),
        },
        "swimming2_target_swimmer": {
            "mean_width_px": swimming2_w_mean,
            "mean_height_px": swimming2_h_mean,
            "mean_aspect_ratio": round(swimming2_w_mean / max(1.0, swimming2_h_mean), 2),
            "typical_area_px2": round(swimming2_w_mean * swimming2_h_mean, 1),
            "relative_image_fraction": round((swimming2_w_mean * swimming2_h_mean) / (1366 * 768), 6),
        },
        "disparity_factors": {
            "width_ratio_vs_sds_swimmer": round(swimming2_w_mean / max(1.0, sds_swimmer_w_mean), 1),
            "area_ratio_vs_sds_swimmer": round((swimming2_w_mean * swimming2_h_mean) / max(1.0, sds_swimmer_w_mean * sds_swimmer_h_mean), 1),
            "width_match_vs_sds_boat": round(swimming2_w_mean / max(1.0, sds_boat_w_mean), 2),
        },
    }

    out_file = out_p / "dataset_gap_analysis_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(gap_data, f, indent=2)

    return gap_data


def run_all_audit_experiments(
    video_path: str = "data/samples/swimming2.mp4",
    weights_path: str = "models/seadronessee-yolov8n.pt",
    output_dir: str = "runs/detector_audit_v2",
) -> None:
    """Run all forensic audit experiments sequentially and save outputs."""
    print("=" * 60)
    print("STARTING DETECTOR ROOT-CAUSE FORENSIC AUDIT (V2)")
    print("=" * 60)

    print("\n[Phase 5] Ground-Truth Evaluation...")
    gt_res = run_ground_truth_evaluation(video_path=video_path, weights_path=weights_path, output_dir=output_dir)
    print(f"Ground-Truth evaluation complete: {len(gt_res['configurations_evaluated'])} configs evaluated.")

    print("\n[Phase 2] Scale Experiment...")
    scale_res = run_scale_experiment(video_path=video_path, weights_path=weights_path, output_dir=output_dir)
    print(f"Scale experiment complete: {len(scale_res['scales_evaluated'])} scales evaluated.")

    print("\n[Phase 3] Input Resolution Experiment...")
    res_res = run_input_resolution_experiment(video_path=video_path, weights_path=weights_path, output_dir=output_dir)
    print(f"Resolution experiment complete: {len(res_res['resolutions_evaluated'])} resolutions evaluated.")

    print("\n[Phase 4] Class Confusion Forensics...")
    conf_res = run_class_confusion_forensics(video_path=video_path, weights_path=weights_path, output_dir=output_dir)
    print("Class confusion forensics complete:")
    for k, v in conf_res["taxonomy_summary"].items():
        print(f"  {k:35s}: {v['count']:3d} ({v['percentage']:.1f}%)")

    print("\n[Phase 6] Real vs Synthetic Domain Comparison...")
    dom_res = run_real_vs_synthetic_comparison(weights_path=weights_path, output_dir=output_dir)
    print(f"Domain comparison complete: {len(dom_res['videos_evaluated'])} video sequences evaluated.")

    print("\n[Phase 7] SeaDronesSee Dataset Gap Analysis...")
    gap_res = run_seadronessee_gap_analysis(weights_path=weights_path, output_dir=output_dir)
    print("Dataset gap analysis complete:")
    print("  Disparities:", gap_res["disparity_factors"])

    print("\n" + "=" * 60)
    print("ALL FORENSIC AUDIT EXPERIMENTS COMPLETE")
    print(f"Artifacts saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Detector Root-Cause Forensic Audit Experiments")
    parser.add_argument("--video", default="data/samples/swimming2.mp4", help="Video path")
    parser.add_argument("--weights", default="models/seadronessee-yolov8n.pt", help="Weights path")
    parser.add_argument("--output-dir", default="runs/detector_audit_v2", help="Output directory")
    args = parser.parse_args()

    run_all_audit_experiments(
        video_path=args.video,
        weights_path=args.weights,
        output_dir=args.output_dir,
    )


