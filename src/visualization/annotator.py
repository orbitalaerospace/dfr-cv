"""Visual annotation utilities for rendered inference outputs."""

from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

from src.domain.enums import TargetClass
from src.schemas.detection import Detection


# BGR Color map for distinct target classes
CLASS_COLORS_BGR: Dict[TargetClass, Tuple[int, int, int]] = {
    TargetClass.PERSON: (255, 165, 0),            # Cyan/Blue-Orange (generic person)
    TargetClass.PERSON_SURFACE: (255, 120, 0),    # Deep Cyan (water surface person)
    TargetClass.SWIMMER: (0, 140, 255),           # Deep Orange
    TargetClass.FLOATER: (0, 215, 255),           # Amber/Gold
    TargetClass.LIFE_JACKET: (203, 192, 255),     # Pink
    TargetClass.WATERCRAFT: (50, 205, 50),        # Lime Green
    TargetClass.UNKNOWN: (160, 160, 160),         # Gray
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
