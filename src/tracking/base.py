"""Abstract base interface for object trackers."""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Optional

from src.schemas.detection import Detection
from src.schemas.tracking import Tracklet


class BaseTracker(ABC):
    """Abstract interface defining the contract for multi-object trackers.

    Trackers consume single-frame `Detection` objects and associate them across
    temporal frame sequences to produce persistent `Tracklet` objects.
    Trackers must remain strictly decoupled from specific detector architectures
    or model weights.
    """

    @property
    @abstractmethod
    def tracker_name(self) -> str:
        """Identifying name of the tracker algorithm and configuration."""
        ...

    @abstractmethod
    def update(
        self,
        detections: List[Detection],
        frame_id: int,
        timestamp: Optional[datetime] = None,
    ) -> List[Tracklet]:
        """Update tracker state with detections observed in the current frame.

        Args:
            detections: List of Detection schemas observed in the current frame.
            frame_id: Monotonic zero-indexed frame sequence identifier.
            timestamp: Observation timestamp. If None, UTC now is assigned.

        Returns:
            List of active Tracklet objects conforming to the domain tracking contract.
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset all internal tracking states, identities, and histories."""
        ...
