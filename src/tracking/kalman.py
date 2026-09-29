"""Kalman Filter for 2D bounding-box spatial motion tracking."""

from typing import Tuple
import numpy as np

from src.schemas.detection import BoundingBox


class KalmanBoxTracker:
    """Kalman filter tracking bounding box state in image pixel space.

    State representation:
        x = [center_x, center_y, aspect_ratio, height, v_x, v_y, v_a, v_h]^T
    where aspect_ratio = width / height.

    Measurement representation:
        z = [center_x, center_y, aspect_ratio, height]^T
    """

    def __init__(
        self,
        std_weight_position: float = 1.0 / 20.0,
        std_weight_velocity: float = 1.0 / 160.0,
    ) -> None:
        self._std_weight_position = std_weight_position
        self._std_weight_velocity = std_weight_velocity

        # State transition matrix F (8x8 constant velocity model)
        self._motion_mat = np.eye(8, 8, dtype=np.float32)
        for i in range(4):
            self._motion_mat[i, i + 4] = 1.0

        # Measurement projection matrix H (4x8)
        self._update_mat = np.eye(4, 8, dtype=np.float32)

    def initiate(self, measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create new track state distribution from an initial 4-element observation.

        Args:
            measurement: Array of shape (4,) containing [cx, cy, a, h].

        Returns:
            Tuple of (mean, covariance):
                mean: State vector of shape (8,)
                covariance: Covariance matrix of shape (8, 8)
        """
        mean_pos = measurement.copy()
        mean_vel = np.zeros_like(mean_pos)
        mean = np.concatenate([mean_pos, mean_vel], axis=0)

        h = measurement[3]
        std = [
            2.0 * self._std_weight_position * h,
            2.0 * self._std_weight_position * h,
            1e-2,
            2.0 * self._std_weight_position * h,
            10.0 * self._std_weight_velocity * h,
            10.0 * self._std_weight_velocity * h,
            1e-5,
            10.0 * self._std_weight_velocity * h,
        ]
        covariance = np.diag(np.square(std)).astype(np.float32)
        return mean, covariance

    def predict(self, mean: np.ndarray, covariance: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Propagate state and uncertainty forward by one discrete time step.

        Args:
            mean: State vector of shape (8,)
            covariance: State covariance matrix of shape (8, 8)

        Returns:
            Tuple of updated (mean, covariance)
        """
        h = mean[3]
        std_pos = [
            self._std_weight_position * h,
            self._std_weight_position * h,
            1e-2,
            self._std_weight_position * h,
        ]
        std_vel = [
            self._std_weight_velocity * h,
            self._std_weight_velocity * h,
            1e-5,
            self._std_weight_velocity * h,
        ]
        q = np.diag(np.square(np.concatenate([std_pos, std_vel], axis=0))).astype(np.float32)

        mean = np.dot(self._motion_mat, mean)
        covariance = np.linalg.multi_dot([self._motion_mat, covariance, self._motion_mat.T]) + q

        # Ensure covariance symmetry
        covariance = 0.5 * (covariance + covariance.T)
        return mean, covariance

    def project(self, mean: np.ndarray, covariance: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Project state distribution into measurement space (4D).

        Args:
            mean: State vector of shape (8,)
            covariance: State covariance matrix of shape (8, 8)

        Returns:
            Tuple of (projected_mean, projected_covariance) in measurement space.
        """
        h = mean[3]
        std = [
            self._std_weight_position * h,
            self._std_weight_position * h,
            1e-1,
            self._std_weight_position * h,
        ]
        r = np.diag(np.square(std)).astype(np.float32)

        projected_mean = np.dot(self._update_mat, mean)
        projected_cov = np.linalg.multi_dot([self._update_mat, covariance, self._update_mat.T]) + r
        return projected_mean, projected_cov

    def update(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurement: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Update state distribution using an incoming 4D measurement vector.

        Args:
            mean: Predicted state vector of shape (8,)
            covariance: Predicted covariance matrix of shape (8, 8)
            measurement: Observed measurement vector of shape (4,)

        Returns:
            Tuple of posterior (mean, covariance).
        """
        projected_mean, projected_cov = self.project(mean, covariance)

        # Innovation
        innovation = measurement - projected_mean

        # Kalman Gain: K = P * H^T * S^-1
        # Equivalent to K = (S^-1 * (H * P))^T
        hp = np.dot(self._update_mat, covariance)  # (4, 8)
        kalman_gain = np.linalg.solve(projected_cov, hp).T  # (8, 4)

        new_mean = mean + np.dot(kalman_gain, innovation)
        new_cov = covariance - np.linalg.multi_dot([kalman_gain, projected_cov, kalman_gain.T])

        # Ensure covariance symmetry
        new_cov = 0.5 * (new_cov + new_cov.T)
        return new_mean, new_cov


def bbox_to_z(bbox: BoundingBox) -> np.ndarray:
    """Convert BoundingBox into Kalman measurement format [cx, cy, a, h]."""
    w = bbox.width
    h = bbox.height
    cx = bbox.x1 + w / 2.0
    cy = bbox.y1 + h / 2.0
    a = w / max(1e-6, h)
    return np.array([cx, cy, a, h], dtype=np.float32)


def z_to_xyxy(z: np.ndarray) -> Tuple[float, float, float, float]:
    """Convert [cx, cy, a, h] state into (x1, y1, x2, y2) coordinates."""
    cx, cy, a, h = float(z[0]), float(z[1]), float(z[2]), float(z[3])
    w = max(1.0, a * h)
    h = max(1.0, h)
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    return x1, y1, x2, y2


def compute_iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Compute pairwise Intersection-over-Union (IoU) matrix.

    Args:
        boxes_a: Array of shape (N, 4) in xyxy format.
        boxes_b: Array of shape (M, 4) in xyxy format.

    Returns:
        Array of shape (N, M) with IoU overlap values in [0.0, 1.0].
    """
    if boxes_a.size == 0 or boxes_b.size == 0:
        return np.zeros((boxes_a.shape[0], boxes_b.shape[0]), dtype=np.float32)

    # Compute intersection
    x1 = np.maximum(boxes_a[:, None, 0], boxes_b[None, :, 0])
    y1 = np.maximum(boxes_a[:, None, 1], boxes_b[None, :, 1])
    x2 = np.minimum(boxes_a[:, None, 2], boxes_b[None, :, 2])
    y2 = np.minimum(boxes_a[:, None, 3], boxes_b[None, :, 3])

    intersection_w = np.maximum(0.0, x2 - x1)
    intersection_h = np.maximum(0.0, y2 - y1)
    intersection = intersection_w * intersection_h

    # Compute union
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    union = area_a[:, None] + area_b[None, :] - intersection

    return np.where(union > 0.0, intersection / np.maximum(1e-6, union), 0.0).astype(np.float32)
