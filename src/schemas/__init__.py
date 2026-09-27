"""Schema definitions and data contracts for perception modules."""

from src.schemas.detection import BoundingBox, Detection
from src.schemas.incident import IncidentAlert
from src.schemas.serialization import from_dict, from_json, to_dict, to_json
from src.schemas.telemetry import UavTelemetry
from src.schemas.tracking import Tracklet

__all__ = [
    "BoundingBox",
    "Detection",
    "IncidentAlert",
    "Tracklet",
    "UavTelemetry",
    "from_dict",
    "from_json",
    "to_dict",
    "to_json",
]
