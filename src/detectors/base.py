"""Abstract base interface for object detector adapters."""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, List, Optional
import numpy as np

from src.domain.enums import TargetClass
from src.schemas.detection import Detection


class BaseDetector(ABC):
    """Abstract interface defining the contract for spatial object detectors.

    Downstream perception pipelines consume `Detection` schemas produced by
    implementations of this class, decoupling downstream logic from specific ML
    frameworks (Ultralytics, ONNX Runtime, TensorRT, TorchScript).
    """

    @property
    @abstractmethod
    def detector_name(self) -> str:
        """Identifying name of the detector model and configuration."""
        ...

    @property
    @abstractmethod
    def target_classes(self) -> List[TargetClass]:
        """List of canonical target classes this detector is configured to report."""
        ...

    @abstractmethod
    def detect(
        self,
        image: np.ndarray,
        frame_id: int = 0,
        timestamp: Optional[datetime] = None,
    ) -> List[Detection]:
        """Execute spatial inference on an input BGR or RGB image array.

        Args:
            image: Image array of shape (H, W, 3).
            frame_id: Zero-indexed video frame counter.
            timestamp: Observation timestamp. If None, UTC now is assigned.

        Returns:
            List of validated, immutable Detection objects conforming to M1 schema.
        """
        ...
