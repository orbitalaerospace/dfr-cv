"""Canonical domain enumerations for aerial perception system.

These string-backed enumerations define the core domain ontology
and lifecycle states. They are intentionally decoupled from any
machine learning frameworks or dataset-specific class IDs.
"""

from enum import Enum


class TargetClass(str, Enum):
    """Canonical classes of objects relevant to aerial search and rescue."""

    PERSON = "person"
    PERSON_SURFACE = "person_surface"
    SWIMMER = "swimmer"
    FLOATER = "floater"
    LIFE_JACKET = "life_jacket"
    LIFE_SAVING_APPLIANCE = "life_saving_appliance"
    BUOY = "buoy"
    WATERCRAFT = "watercraft"
    UNKNOWN = "unknown"


class TrackState(str, Enum):
    """Lifecycle states of a tracked object across time."""

    NEW = "new"
    TRACKED = "tracked"
    COASTING = "coasting"
    LOST = "lost"


class IncidentStatus(str, Enum):
    """Lifecycle states of an operational aerial incident."""

    DETECTED = "detected"
    CANDIDATE = "candidate"
    CONFIRMED = "confirmed"
    RESOLVED = "resolved"
    REJECTED = "rejected"


class AlertSeverity(str, Enum):
    """Severity ratings for dispatched incident alerts."""

    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
