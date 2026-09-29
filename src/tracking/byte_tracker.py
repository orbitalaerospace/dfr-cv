"""ByteTrack multi-object tracking implementation conforming to BaseTracker."""

from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment

from src.domain.enums import TargetClass, TrackState
from src.schemas.detection import BoundingBox, Detection
from src.schemas.tracking import Tracklet
from src.tracking.base import BaseTracker
from src.tracking.kalman import KalmanBoxTracker, bbox_to_z, compute_iou_matrix, z_to_xyxy


class _Track:
    """Internal mutable state representation for a single tracked target."""

    def __init__(
        self,
        track_id: int,
        detection: Detection,
        kalman: KalmanBoxTracker,
        min_hits: int = 3,
    ) -> None:
        self.track_id = track_id
        self.target_class = detection.target_class
        self.score = float(detection.confidence)
        self.kalman = kalman
        self.min_hits = min_hits

        # Initialize Kalman state
        z = bbox_to_z(detection.bbox)
        self.mean, self.covariance = self.kalman.initiate(z)

        # Lifecycle metrics
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.state = TrackState.TRACKED if min_hits <= 1 else TrackState.NEW

        # Temporal timestamps & observation records
        self.first_seen_timestamp = detection.timestamp
        self.last_seen_timestamp = detection.timestamp
        self.last_detection = detection
        self.history: List[Detection] = [detection]

    def predict(self) -> None:
        """Propagate state forward one discrete time step."""
        self.mean, self.covariance = self.kalman.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1

    def update(self, detection: Detection) -> None:
        """Update track with a newly matched detection."""
        z = bbox_to_z(detection.bbox)
        self.mean, self.covariance = self.kalman.update(self.mean, self.covariance, z)

        self.target_class = detection.target_class
        self.score = float(detection.confidence)
        self.last_detection = detection
        self.history.append(detection)

        self.hits += 1
        self.time_since_update = 0
        self.last_seen_timestamp = detection.timestamp

        if self.state == TrackState.NEW and self.hits >= self.min_hits:
            self.state = TrackState.TRACKED
        elif self.state == TrackState.COASTING:
            self.state = TrackState.TRACKED

    def mark_missed(self, max_lost: int) -> None:
        """Update track lifecycle state when no detection matches in the current frame."""
        if self.state == TrackState.NEW:
            self.state = TrackState.LOST
        elif self.time_since_update > max_lost:
            self.state = TrackState.LOST
        else:
            self.state = TrackState.COASTING

    def to_bounding_box(self) -> BoundingBox:
        """Generate current spatial BoundingBox from Kalman state estimate."""
        x1, y1, x2, y2 = z_to_xyxy(self.mean[:4])
        # Clamp to valid non-negative coordinates satisfying x2 > x1 and y2 > y1
        x1 = max(0.0, x1)
        y1 = max(0.0, y1)
        x2 = max(x1 + 1.0, x2)
        y2 = max(y1 + 1.0, y2)
        return BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)

    def to_tracklet(self, frame_id: int, timestamp: datetime) -> Tracklet:
        """Construct an immutable domain Tracklet object representing current state."""
        if self.state == TrackState.COASTING:
            # During coasting, generate a predicted detection representing the projected position
            # with decayed confidence and explicit detector identification
            pred_bbox = self.to_bounding_box()
            decay = max(0.05, self.score * (0.95 ** self.time_since_update))
            current_det = Detection(
                bbox=pred_bbox,
                target_class=self.target_class,
                confidence=round(decay, 3),
                frame_id=frame_id,
                timestamp=timestamp,
                detector_name="kalman_coasting",
            )
        else:
            current_det = self.last_detection

        return Tracklet(
            track_id=self.track_id,
            state=self.state,
            current_detection=current_det,
            first_seen_timestamp=self.first_seen_timestamp,
            last_seen_timestamp=self.last_seen_timestamp,
            observation_history=tuple(self.history),
        )


class ByteTracker(BaseTracker):
    """Multi-object tracker implementing the ByteTrack association algorithm.

    Associates both high-confidence and low-confidence detections using a 2-stage
    linear sum assignment on IoU distance to bridge temporary occlusions and wave
    crest submersions without losing identity.
    """

    def __init__(
        self,
        high_threshold: float = 0.5,
        low_threshold: float = 0.1,
        match_threshold_high: float = 0.8,
        match_threshold_low: float = 0.5,
        max_lost: int = 30,
        min_hits: int = 3,
        enforce_class_match: bool = True,
        emit_coasting: bool = True,
        emit_unconfirmed: bool = True,
        tracker_name: str = "bytetrack",
    ) -> None:
        """Initialize ByteTracker with configurable lifecycle and matching parameters.

        Args:
            high_threshold: Minimum confidence threshold for Stage 1 primary matching.
            low_threshold: Minimum confidence threshold for Stage 2 recovery matching.
            match_threshold_high: Maximum IoU distance (1 - IoU) allowed in Stage 1.
            match_threshold_low: Maximum IoU distance allowed in Stage 2.
            max_lost: Maximum consecutive frames a track can coast before termination.
            min_hits: Minimum detection matches required to confirm a track as TRACKED.
            enforce_class_match: Whether to prevent cross-class associations.
            emit_coasting: Whether to include COASTING tracklets in public update output.
            emit_unconfirmed: Whether to include NEW tracklets in public update output.
            tracker_name: Identifying name of the tracker configuration.
        """
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.match_threshold_high = match_threshold_high
        self.match_threshold_low = match_threshold_low
        self.max_lost = max_lost
        self.min_hits = min_hits
        self.enforce_class_match = enforce_class_match
        self.emit_coasting = emit_coasting
        self.emit_unconfirmed = emit_unconfirmed
        self._tracker_name = tracker_name

        self._kalman = KalmanBoxTracker()
        self._next_id = 1
        self._tracked_tracks: List[_Track] = []
        self._coasting_tracks: List[_Track] = []

    @property
    def tracker_name(self) -> str:
        return self._tracker_name

    def reset(self) -> None:
        """Reset all active tracks and reset ID counter."""
        self._next_id = 1
        self._tracked_tracks.clear()
        self._coasting_tracks.clear()

    def update(
        self,
        detections: List[Detection],
        frame_id: int,
        timestamp: Optional[datetime] = None,
    ) -> List[Tracklet]:
        """Update tracker state with detections observed in the current frame."""
        now = timestamp if timestamp is not None else datetime.now(timezone.utc)

        # 1. Partition detections by confidence
        dets_high: List[Detection] = []
        dets_low: List[Detection] = []
        for det in detections:
            if det.confidence >= self.high_threshold:
                dets_high.append(det)
            elif det.confidence >= self.low_threshold:
                dets_low.append(det)

        # 2. Predict next spatial state for all existing tracks
        active_tracks = self._tracked_tracks + self._coasting_tracks
        for track in active_tracks:
            track.predict()

        # 3. Stage 1: Associate active tracks with high-confidence detections
        matched_tracks, unmatched_tracks_s1, unmatched_dets_s1 = self._associate(
            tracks=active_tracks,
            detections=dets_high,
            cost_threshold=self.match_threshold_high,
        )

        # Apply Stage 1 updates
        for track, det in matched_tracks:
            track.update(det)

        # 4. Stage 2: Associate remaining active tracks with low-confidence detections
        # In ByteTrack, Stage 2 matches only tracks that were TRACKED (skip NEW unconfirmed tracks)
        tracked_pool = [t for t in unmatched_tracks_s1 if t.state == TrackState.TRACKED]
        other_unmatched = [t for t in unmatched_tracks_s1 if t.state != TrackState.TRACKED]

        matched_tracks_s2, unmatched_tracks_s2, _ = self._associate(
            tracks=tracked_pool,
            detections=dets_low,
            cost_threshold=self.match_threshold_low,
        )

        # Apply Stage 2 updates
        for track, det in matched_tracks_s2:
            track.update(det)

        # All unmatched tracks from Stage 1 & Stage 2
        all_unmatched_tracks = other_unmatched + unmatched_tracks_s2
        for track in all_unmatched_tracks:
            track.mark_missed(self.max_lost)

        # 5. Initialize new tracks from remaining unmatched high-confidence detections
        new_tracks: List[_Track] = []
        for det in unmatched_dets_s1:
            new_track = _Track(
                track_id=self._next_id,
                detection=det,
                kalman=self._kalman,
                min_hits=self.min_hits,
            )
            self._next_id += 1
            new_tracks.append(new_track)

        # 6. Re-aggregate tracks into tracked and coasting pools
        all_candidate_tracks = [t for t, _ in matched_tracks] + [t for t, _ in matched_tracks_s2] + all_unmatched_tracks + new_tracks
        self._tracked_tracks = [t for t in all_candidate_tracks if t.state == TrackState.TRACKED or (t.state == TrackState.NEW and t.hits >= self.min_hits)]
        self._coasting_tracks = [t for t in all_candidate_tracks if t.state == TrackState.COASTING]

        # Handle newly created unconfirmed tracks if min_hits > 1
        unconfirmed_tracks = [t for t in all_candidate_tracks if t.state == TrackState.NEW and t.hits < self.min_hits]
        self._tracked_tracks.extend(unconfirmed_tracks)

        # 7. Construct output Tracklet instances
        output_tracklets: List[Tracklet] = []
        for track in self._tracked_tracks + self._coasting_tracks:
            if track.state == TrackState.LOST:
                continue
            if track.state == TrackState.COASTING and not self.emit_coasting:
                continue
            if track.state == TrackState.NEW and not self.emit_unconfirmed:
                continue

            tracklet = track.to_tracklet(frame_id=frame_id, timestamp=now)
            output_tracklets.append(tracklet)

        # Return sorted by track_id for deterministic downstream consumption
        output_tracklets.sort(key=lambda t: t.track_id)
        return output_tracklets

    def _associate(
        self,
        tracks: List[_Track],
        detections: List[Detection],
        cost_threshold: float,
    ) -> Tuple[List[Tuple[_Track, Detection]], List[_Track], List[Detection]]:
        """Perform optimal bipartite matching using IoU distance and Hungarian assignment."""
        if len(tracks) == 0 or len(detections) == 0:
            return [], list(tracks), list(detections)

        track_boxes = np.array([t.to_bounding_box().as_xyxy() for t in tracks], dtype=np.float32)
        det_boxes = np.array([d.bbox.as_xyxy() for d in detections], dtype=np.float32)

        # Cost matrix: 1.0 - IoU
        iou_matrix = compute_iou_matrix(track_boxes, det_boxes)
        cost_matrix = 1.0 - iou_matrix

        # Penalize cross-class matches if class matching is enforced
        if self.enforce_class_match:
            for i, track in enumerate(tracks):
                for j, det in enumerate(detections):
                    if track.target_class != det.target_class:
                        cost_matrix[i, j] = 1.0

        # Solve assignment problem
        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        matched_pairs: List[Tuple[_Track, Detection]] = []
        unmatched_track_indices = set(range(len(tracks)))
        unmatched_det_indices = set(range(len(detections)))

        for r, c in zip(row_indices, col_indices):
            if cost_matrix[r, c] <= cost_threshold:
                matched_pairs.append((tracks[r], detections[c]))
                unmatched_track_indices.discard(r)
                unmatched_det_indices.discard(c)

        unmatched_tracks = [tracks[i] for i in sorted(unmatched_track_indices)]
        unmatched_dets = [detections[j] for j in sorted(unmatched_det_indices)]

        return matched_pairs, unmatched_tracks, unmatched_dets
