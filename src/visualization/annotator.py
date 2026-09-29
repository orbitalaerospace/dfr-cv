"""Visual annotation utilities for rendered inference outputs."""

from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

from src.domain.enums import BehaviorState, TargetClass, TrackState
from src.schemas.behavior import BehaviorAssessment
from src.schemas.detection import Detection
from src.schemas.tracking import Tracklet


# BGR Color map for distinct target classes
CLASS_COLORS_BGR: Dict[TargetClass, Tuple[int, int, int]] = {
    TargetClass.PERSON: (255, 165, 0),            # Cyan/Blue-Orange (generic person)
    TargetClass.PERSON_SURFACE: (255, 120, 0),    # Deep Cyan (water surface person)
    TargetClass.SWIMMER: (0, 140, 255),           # Deep Orange
    TargetClass.FLOATER: (0, 215, 255),           # Amber/Gold
    TargetClass.LIFE_JACKET: (203, 192, 255),     # Pink
    TargetClass.LIFE_SAVING_APPLIANCE: (180, 105, 255),  # Violet
    TargetClass.BUOY: (0, 255, 255),              # Yellow
    TargetClass.WATERCRAFT: (50, 205, 50),        # Lime Green
    TargetClass.UNKNOWN: (160, 160, 160),         # Gray
}

# BGR Color map for behavior states
BEHAVIOR_COLORS_BGR: Dict[BehaviorState, Tuple[int, int, int]] = {
    BehaviorState.DISTRESS_CONFIRMED: (0, 0, 235),      # Vivid Crimson Red
    BehaviorState.DISTRESS_CANDIDATE: (0, 140, 255),    # Vibrant Amber / Orange
    BehaviorState.NORMAL: (34, 185, 34),                # Forest / Sea Green
    BehaviorState.UNKNOWN: (170, 170, 170),             # Neutral Slate Gray
}



def draw_detections(
    image: np.ndarray,
    detections: List[Detection],
    show_labels: bool = True,
    show_confidence: bool = True,
    line_thickness: int = 2,
    header_text: Optional[str] = None,
) -> np.ndarray:
    """Draw bounding boxes and class banners onto an image array.

    Args:
        image: Original BGR image array (H, W, 3).
        detections: List of Detection schema objects to render.
        show_labels: Whether to render class names.
        show_confidence: Whether to append confidence scores.
        line_thickness: Bounding box outline width in pixels.
        header_text: Optional top-left status banner string.

    Returns:
        Annotated BGR copy of the image.
    """
    annotated = image.copy()

    for det in detections:
        box = det.bbox
        x1, y1, x2, y2 = int(box.x1), int(box.y1), int(box.x2), int(box.y2)
        color = CLASS_COLORS_BGR.get(det.target_class, (0, 255, 0))

        # Draw main bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, line_thickness)

        if show_labels:
            label_parts = [det.target_class.value]
            if show_confidence:
                label_parts.append(f"{det.confidence:.2f}")
            label = " ".join(label_parts)

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_thickness = 1
            (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)

            # Draw filled background banner for text readability
            banner_y1 = max(0, y1 - text_h - baseline - 4)
            banner_y2 = y1
            banner_x2 = min(annotated.shape[1], x1 + text_w + 6)
            cv2.rectangle(annotated, (x1, banner_y1), (banner_x2, banner_y2), color, -1)

            # Draw white text over color banner
            text_y = y1 - baseline - 2
            cv2.putText(
                annotated,
                label,
                (x1 + 3, text_y),
                font,
                font_scale,
                (255, 255, 255),
                font_thickness,
                lineType=cv2.LINE_AA,
            )

    if header_text:
        # Top status bar overlay
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 30), (0, 0, 0), -1)
        cv2.putText(
            annotated,
            header_text,
            (10, 20),
            font,
            0.55,
            (0, 255, 255),
            1,
            lineType=cv2.LINE_AA,
        )

    return annotated


def draw_tracks(
    image: np.ndarray,
    tracklets: List[Tracklet],
    behaviors: Optional[Dict[int, BehaviorAssessment]] = None,
    show_labels: bool = True,
    show_confidence: bool = True,
    show_history_trail: bool = True,
    max_trail_points: int = 30,
    line_thickness: int = 2,
    header_text: Optional[str] = None,
) -> np.ndarray:
    """Draw tracking bounding boxes, persistent IDs, behavior state, and trajectory trails.

    Args:
        image: Original BGR image array (H, W, 3).
        tracklets: List of Tracklet schema objects to render.
        behaviors: Optional mapping from track_id to BehaviorAssessment.
        show_labels: Whether to render class names and track IDs.
        show_confidence: Whether to append confidence scores.
        show_history_trail: Whether to draw historical trajectory breadcrumbs.
        max_trail_points: Maximum number of historical trajectory points to render.
        line_thickness: Bounding box outline width in pixels.
        header_text: Optional top status bar banner string.

    Returns:
        Annotated BGR copy of the image.
    """
    annotated = image.copy()

    for trk in tracklets:
        det = trk.current_detection
        box = det.bbox
        x1, y1, x2, y2 = int(box.x1), int(box.y1), int(box.x2), int(box.y2)
        base_color = CLASS_COLORS_BGR.get(det.target_class, (0, 255, 0))

        # Check for behavioral assessment
        assessment = behaviors.get(trk.track_id) if behaviors else None
        is_coasting = trk.state == TrackState.COASTING

        # Adjust visual representation for COASTING or BEHAVIOR state
        if is_coasting:
            color = (180, 180, 180)  # Muted silver/gray for coasting prediction
            box_thickness = max(1, line_thickness - 1)
        elif assessment and det.target_class in (TargetClass.SWIMMER, TargetClass.PERSON, TargetClass.PERSON_SURFACE):
            color = BEHAVIOR_COLORS_BGR.get(assessment.state, base_color)
            box_thickness = max(2, line_thickness + 1) if assessment.state == BehaviorState.DISTRESS_CONFIRMED else line_thickness
        else:
            color = base_color
            box_thickness = line_thickness

        # Draw trajectory breadcrumbs/trail
        if show_history_trail and len(trk.observation_history) > 1:
            recent_obs = trk.observation_history[-max_trail_points:]
            for i in range(len(recent_obs) - 1):
                pt_a = (int(recent_obs[i].bbox.center[0]), int(recent_obs[i].bbox.center[1]))
                pt_b = (int(recent_obs[i + 1].bbox.center[0]), int(recent_obs[i + 1].bbox.center[1]))
                cv2.line(annotated, pt_a, pt_b, color, 1, lineType=cv2.LINE_AA)

            # Draw latest center dot
            center = (int(box.center[0]), int(box.center[1]))
            cv2.circle(annotated, center, 3, color, -1)

        # Draw main bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, box_thickness)

        if show_labels:
            # Construct multi-line banner labels
            label_parts = [f"{det.target_class.value.upper()} #{trk.track_id:02d}"]
            if is_coasting:
                label_parts.append("[COAST]")
            elif show_confidence:
                label_parts.append(f"({det.confidence:.2f})")
            line1 = " ".join(label_parts)

            lines_to_render = [line1]

            # Append behavioral assessment line if available
            if assessment and det.target_class in (TargetClass.SWIMMER, TargetClass.PERSON, TargetClass.PERSON_SURFACE):
                if assessment.state == BehaviorState.DISTRESS_CONFIRMED:
                    line2 = f"BEHAVIOR: DISTRESS ({assessment.distress_score:.2f})"
                elif assessment.state == BehaviorState.DISTRESS_CANDIDATE:
                    line2 = f"BEHAVIOR: CANDIDATE ({assessment.distress_score:.2f})"
                elif assessment.state == BehaviorState.NORMAL:
                    line2 = f"BEHAVIOR: NORMAL ({assessment.distress_score:.2f})"
                else:
                    line2 = "BEHAVIOR: UNKNOWN"
                lines_to_render.append(line2)

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.45
            font_thickness = 1

            # Render stacked label banners from bottom up above box
            curr_y = y1
            for text_line in reversed(lines_to_render):
                (text_w, text_h), baseline = cv2.getTextSize(text_line, font, font_scale, font_thickness)
                banner_y1 = max(0, curr_y - text_h - baseline - 4)
                banner_y2 = curr_y
                banner_x2 = min(annotated.shape[1], x1 + text_w + 6)
                cv2.rectangle(annotated, (x1, banner_y1), (banner_x2, banner_y2), color, -1)

                text_y = curr_y - baseline - 2
                cv2.putText(
                    annotated,
                    text_line,
                    (x1 + 3, text_y),
                    font,
                    font_scale,
                    (255, 255, 255),
                    font_thickness,
                    lineType=cv2.LINE_AA,
                )
                curr_y = banner_y1 - 1


    if header_text:
        # Top status bar overlay
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 30), (0, 0, 0), -1)
        cv2.putText(
            annotated,
            header_text,
            (10, 20),
            font,
            0.55,
            (0, 255, 255),
            1,
            lineType=cv2.LINE_AA,
        )

    return annotated
