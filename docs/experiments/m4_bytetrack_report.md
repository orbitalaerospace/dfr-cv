# M4 ByteTrack Integration Report

## 1. Architecture

Milestone 4 integrates a detector-agnostic Multi-Object Tracking (MOT) layer into `dfr-cv` using the **ByteTrack** association paradigm.

The perception pipeline is decoupled into explicit, single-responsibility stages:

```
[Sequential Video / Camera Stream]
               │
               ▼
┌──────────────────────────────────────────────┐
│ BaseDetector (UltralyticsDetectorAdapter)    │  ◄── SeaDronesSee YOLOv8n (or COCO YOLO11n)
└──────────────────────────────────────────────┘
               │
               ▼ List[Detection]  (Canonical spatial contracts)
┌──────────────────────────────────────────────┐
│ BaseTracker (ByteTracker)                    │  ◄── Pure NumPy / SciPy implementation
│  - 8D Kalman filter state propagation        │
│  - 2-stage IoU bipartite association         │
│  - Mutable internal state tracking           │
│  - Schema projection to immutable Tracklet   │
└──────────────────────────────────────────────┘
               │
               ▼ List[Tracklet]   (Persistent identities & history)
┌──────────────────────────────────────────────┐
│ Visualization & Downstream Perception        │  ◄── draw_tracks() (ID banners & trails)
└──────────────────────────────────────────────┘
```

### Architectural Guarantees:
- **Detector Agnostic**: The tracker consumes strictly canonical `List[Detection]` schemas. It contains zero imports or dependencies on PyTorch, Ultralytics, YOLO, or SeaDronesSee-specific structures.
- **Contract Conformance**: The tracker emits strictly immutable `List[Tracklet]` objects conforming to `src/schemas/tracking.py`.
- **Zero Heavy Tracking Dependencies**: Operates entirely with `numpy` and `scipy.optimize.linear_sum_assignment`, eliminating fragile C++ compilation dependencies like `lap` or `lapx`.

---

## 2. ByteTrack Implementation

The tracking core is implemented in `src/tracking/byte_tracker.py` and inherits from `src/tracking/base.py:BaseTracker`.

### Two-Stage Association Logic:
1. **Confidence Partitioning**:
   Incoming detections are partitioned into two priority sets:
   - High-confidence set: $D_{\text{high}} = \{ d \in \text{detections} \mid d.\text{confidence} \ge \tau_{\text{high}} \}$
   - Low-confidence set: $D_{\text{low}} = \{ d \in \text{detections} \mid \tau_{\text{low}} \le d.\text{confidence} < \tau_{\text{high}} \}$
2. **Kalman Prediction**:
   All active tracks in `self._tracked_tracks` and `self._coasting_tracks` propagate their mean and covariance forward by one discrete time step.
3. **Stage 1 (Primary Association)**:
   Active tracks are matched against $D_{\text{high}}$ using an IoU distance cost matrix ($C = 1 - \text{IoU}$).
   Matching is solved optimally via `scipy.optimize.linear_sum_assignment`. Pairs exceeding `match_threshold_high` (default 0.8, i.e., $\text{IoU} < 0.2$) are rejected.
4. **Stage 2 (Recovery Association)**:
   Tracks that remained unmatched in Stage 1 and are confirmed `TRACKED` are matched against $D_{\text{low}}$.
   This stage recovers occluded, blurred, or submerged swimmers who suffer momentary confidence dips without creating new identities. Pairs exceeding `match_threshold_low` (default 0.5) are rejected.
5. **Lifecycle Progression**:
   - Unmatched tracks increment their `time_since_update` counter and transition to `COASTING` or `LOST`.
   - Unmatched high-confidence detections initialize new tracks with unique IDs.
6. **Public Schema Projection**:
   Active tracks are projected into immutable `Tracklet` objects.

---

## 3. Kalman State Representation

Implemented in `src/tracking/kalman.py:KalmanBoxTracker`:

### State Vector ($\mathbf{x} \in \mathbb{R}^8$):
$$\mathbf{x} = [x_c, y_c, a, h, v_x, v_y, v_a, v_h]^T$$
- $(x_c, y_c)$: Bounding box center coordinates in image pixel space.
- $a = w / h$: Bounding box aspect ratio.
- $h$: Bounding box height.
- $(v_x, v_y, v_a, v_h)$: Corresponding discrete velocities.

### Measurement Vector ($\mathbf{z} \in \mathbb{R}^4$):
$$\mathbf{z} = [x_c, y_c, a, h]^T$$

### Noise Modeling:
Process noise $\mathbf{Q}$ and measurement noise $\mathbf{R}$ scale dynamically with target bounding box height $h$:
- Position uncertainty: $\sigma_p = \frac{1}{20} \cdot h$
- Velocity uncertainty: $\sigma_v = \frac{1}{160} \cdot h$
- Aspect ratio variation is constrained with small fixed noise to maintain spatial stability during wave turbulence.

---

## 4. Association Strategy

- **Metric**: Bounding box Intersection-over-Union (IoU) distance:
  $$\text{Cost}(i, j) = 1.0 - \text{IoU}(\text{bbox}_i, \text{bbox}_j)$$
- **Optimization**: Global cost minimization via Kuhn-Munkres (Hungarian algorithm) implemented in `scipy.optimize.linear_sum_assignment`.
- **Class Isolation Policy (`enforce_class_match=True`)**:
  Pairs with discordant `TargetClass` (e.g., matching a `SWIMMER` track with a `WATERCRAFT` detection) are assigned cost $1.0$ (infinite penalty). This guarantees that passing watercraft never steal or corrupt a swimmer's track identity.

---

## 5. Track Lifecycle

Target tracks transition through the canonical `TrackState` enum (`src/domain/enums.py`):

```
       [Unmatched High-Conf Detection]
                      │
                      ▼
               TrackState.NEW  (hits = 1)
                      │
        ┌─────────────┴─────────────┐
 (hits >= min_hits)           (missed)
        │                           │
        ▼                           ▼
TrackState.TRACKED ◄───┐     TrackState.LOST
        │              │       (Pruned)
     (missed)      (matched)
        │              │
        ▼              │
TrackState.COASTING ───┘
        │
(time_since_update > max_lost)
        │
        ▼
TrackState.LOST (Pruned)
```

### Coasting State & Schema Integrity:
- The domain `Tracklet` schema requires `current_detection: Detection` (non-nullable).
- When a target enters `COASTING`, no physical camera detection occurred in that frame.
- To maintain complete contract validity without fabricating data, the tracker projects the Kalman filter state into a predicted `BoundingBox`, assigns a decayed confidence, and marks `detector_name="kalman_coasting"`.
- This informs downstream reasoning that the bounding box is a motion extrapolation, not an unconfirmed physical observation.

---

## 6. Configuration Parameters

The tracker exposes configurable parameters with empirical defaults tailored for aerial maritime footage:

| Parameter | Default | Function |
|:---|:---:|:---|
| `high_threshold` | `0.40` | Lower bound for primary Stage 1 detection matching. |
| `low_threshold` | `0.10` | Lower bound for secondary recovery matching of submerged targets. |
| `match_threshold_high` | `0.80` | Maximum IoU distance ($1 - \text{IoU}$) for Stage 1 (requires $\ge 20\%$ overlap). |
| `match_threshold_low` | `0.50` | Maximum IoU distance for Stage 2 recovery (requires $\ge 50\%$ overlap). |
| `max_lost` | `30` | Number of consecutive coasting frames before pruning (3.0s at 10 FPS). |
| `min_hits` | `3` | Consecutive detection matches required to confirm a track as `TRACKED`. |
| `enforce_class_match` | `True` | Prevents cross-class identity swaps (swimmer vs boat). |
| `emit_coasting` | `True` | Emits coasting targets with projected bounding boxes for continuous downstream tracking. |
| `emit_unconfirmed` | `True` | Emits newly initiated tracks with state `TrackState.NEW`. |

---

## 7. Unit-Test Results

A comprehensive unit test suite was implemented in `tests/unit/test_bytetrack.py` and `tests/unit/test_track_pipeline.py`. All tests run deterministically with synthetic detections without requiring GPU access or neural network inference.

```text
tests/unit/test_bytetrack.py::test_1_tracker_initialization PASSED
tests/unit/test_bytetrack.py::test_2_first_detection_creates_track PASSED
tests/unit/test_bytetrack.py::test_3_track_stable_id_across_consecutive_frames PASSED
tests/unit/test_bytetrack.py::test_4_two_detections_create_two_distinct_tracks PASSED
tests/unit/test_bytetrack.py::test_5_tracks_remain_associated_when_objects_move_smoothly PASSED
tests/unit/test_bytetrack.py::test_6_high_confidence_association PASSED
tests/unit/test_bytetrack.py::test_7_low_confidence_second_stage_association PASSED
tests/unit/test_bytetrack.py::test_8_temporary_detection_dropout_produces_coasting PASSED
tests/unit/test_bytetrack.py::test_9_detection_returning_after_dropout_reconnects_to_same_id PASSED
tests/unit/test_bytetrack.py::test_10_track_pruned_after_max_lost PASSED
tests/unit/test_bytetrack.py::test_11_observation_history_grows_correctly PASSED
tests/unit/test_bytetrack.py::test_12_tracklet_invariants_remain_valid PASSED
tests/unit/test_bytetrack.py::test_13_enforce_class_matching_prevents_swaps PASSED
tests/unit/test_track_pipeline.py::test_detector_to_bytetrack_integration_pipeline PASSED
tests/unit/test_track_pipeline.py::test_real_detector_to_tracker_pipeline PASSED
```

**Full Test Suite Result**: **61 passed in 2.66s** (including all 46 existing tests from previous milestones).

---

## 8. Real-Video Test Results

The full sequential tracking pipeline (`src/pipeline/track_video.py`) was executed on real aerial video using the SeaDronesSee domain detector (`models/seadronessee-yolov8n.pt`) on Apple Silicon MPS:

### Test 1: Existing Aerial Sample (`aerial_water_sample.mp4`)
- **Video Metadata**: 41 frames | $640 \times 360$ | 10.0 FPS | Lake Constance UAV flight
- **Output Artifact**: `runs/m4/seadronessee_tracking_sample.mp4`
- **Total Execution Time**: 1.86 s
- **Effective Pipeline Throughput**: **22.03 FPS**
- **Mean Frame Latency**: **43.10 ms**
  - Spatial Detection Latency: 42.88 ms
  - **ByteTrack Tracking Latency**: **0.22 ms** ($0.5\%$ of total frame time)
- **Total Detections Emitted**: 239
- **Total Unique Tracks Initialized**: 21
  - Swimmer Tracks: 14 (Primary persistent swimmer IDs: `#04`, `#05`, `#06`, `#08`, `#09`)
  - Watercraft Tracks: 7 (Primary boat IDs: `#01`, `#02`, `#03`)
- **Frames with Active Swimmers**: 36 / 41 ($87.8\%$)
- **Detection Gaps Bridged**: 3 instances where submerged/occluded targets coasted and successfully recovered their original ID.
- **Maximum Track Lifespan**: 41 frames (Track `#01` and `#02` tracked continuously across the entire clip).

### Test 2: Unseen Aerial 4K Sequence (`aerial_unseen_sequence.mp4`)
- **Video Metadata**: 20 frames | $3840 \times 2160$ (4K UHD) | 10.0 FPS | SeaDronesSee Validation set (10416–10435)
- **Output Artifact**: `runs/m4/seadronessee_tracking_unseen.mp4`
- **Total Execution Time**: 2.57 s
- **Effective Pipeline Throughput**: **7.77 FPS** (Full 4K un-tiled processing)
- **Mean Frame Latency**: **80.73 ms**
  - Spatial Detection Latency: 80.26 ms
  - **ByteTrack Tracking Latency**: **0.47 ms** ($0.6\%$ of total frame time)
- **Total Detections Emitted**: 229
- **Frames with Active Swimmers**: 20 / 20 ($100.0\%$)
- **Detection Gaps Bridged**: 3

---

## 9. Visual Observations

Inspection of the generated tracking video artifacts reveals:
1. **Identity Stability**:
   - Moving swimmers maintain persistent numeric tags across frames (e.g., `SWIMMER #04`, `SWIMMER #05`).
   - Rather than generating arbitrary new IDs every frame, tracks preserve identity over dozens of frames.
2. **Trajectory Trails**:
   - The historical breadcrumb trail rendered behind each swimmer clearly illustrates directional swimming motion versus localized bobbing.
3. **Occlusion Handling**:
   - When a swimmer dips beneath a wave swell for 1–2 frames, the tracker enters `[COAST]` with a silver bounding box and immediately re-locks onto the original ID once the head resurfaces.
4. **Class Segregation**:
   - Boats and swimmers moving in proximity maintain separate tracks without identity crossover.

---

## 10. Failure Cases & Edge Observations

1. **Transient Wave Noise Near Threshold**:
   - Sporadic single-frame false detections on splashing water occasionally initialize `NEW` tracks. Because `min_hits=3`, these unconfirmed tracks die after 1 frame without corrupting the confirmed swimmer registry.
2. **Close-Proximity Swimmer Clustering**:
   - When two swimmers swim directly alongside each other ($<10$ pixels separation at high altitude), the spatial detector occasionally emits a single merged bounding box covering both individuals. When they separate, one swimmer retains the original ID and the other receives a new ID.
3. **High Drone Attitude Slew**:
   - If the drone gimbal pans rapidly, Kalman constant-velocity assumption undergoes brief drift. Incorporating Camera Motion Compensation (CMC) in future milestones will further strengthen association during aggressive UAV maneuvers.

---

## 11. Performance Analysis

| Pipeline Component | $640 \times 360$ Stream | $3840 \times 2160$ 4K Stream | Compute Hardware |
|:---|:---:|:---:|:---:|
| **YOLO Spatial Detector** | 42.88 ms | 80.26 ms | Apple Silicon MPS (`mps`) |
| **ByteTrack Tracking** | **0.22 ms** | **0.47 ms** | CPU (NumPy / SciPy) |
| **Video Decoding & HUD Overlay** | 0.00 ms (pipelined) | 0.00 ms (pipelined) | CPU (OpenCV) |
| **Total Frame Latency** | **43.10 ms** | **80.73 ms** | Hybrid |
| **Effective Throughput** | **22.03 FPS** | **7.77 FPS** | Real-time capable |

*Takeaway*: The ByteTrack tracking algorithm accounts for less than **$1\%$** of total pipeline latency. The computational bottleneck remains purely within the deep-learning spatial detector.

---

## 12. Known Limitations

1. **2D Pixel-Space Dynamics**:
   Kalman velocities are tracked in 2D image coordinates rather than 3D georeferenced space. Sudden drone acceleration affects bounding box velocity vectors.
2. **No Re-Identification (ReID) Appearance Embeddings**:
   ByteTrack relies purely on high/low confidence IoU geometry. If a target is occluded for $>30$ frames and drifts far from its trajectory, appearance-based matching (e.g. BoT-SORT) would be required for re-acquisition.
3. **Short Test Clips**:
   Validation clips span 20–41 frames. Extended 10-minute patrol footage will be used in future operational milestones.

---

## 13. What M5 Needs from the Tracker

> [!IMPORTANT]
> **SWIMMER detection is NOT drowning detection.**  
> Milestone 4 only provides persistent swimmer identities and temporal observation history.

Milestone 5 (Temporal Incident Reasoning) requires the following specific capabilities delivered by this tracking layer:
1. **Chronological Observation History**:
   Each `Tracklet` provides `observation_history: Tuple[Detection, ...]`, giving M5 access to sequential bounding box centers, aspect ratios, and confidence progressions over time.
2. **Kinematic Feature Extraction**:
   Using the tracklet history, M5 will compute:
   - Net displacement vs total distance traveled (straight-line swimming vs stationary flailing).
   - Horizontal velocity variance (smooth stroke propulsion vs stationary immersion).
   - Bounding box aspect-ratio oscillation frequency (vertical bobbing vs horizontal stroke posture).
   - Submersion / coasting frequency (periodic disappearance under water surface).
3. **Incident State Machine Integration**:
   M5 will feed track kinematics into an incident state machine (`NEW -> CANDIDATE -> CONFIRMED`), escalating from `TargetClass.SWIMMER` to `TargetClass.DISTRESS` only when temporal distress criteria are met.

---

## 14. Acceptance Criteria Verification

- [x] **ByteTrack implemented under `src/tracking/`**: `base.py`, `kalman.py`, `byte_tracker.py`, `__init__.py`.
- [x] **Tracker is detector-agnostic**: Operates strictly on `List[Detection]`.
- [x] **Existing contracts reused**: Uses `BoundingBox`, `Detection`, `Tracklet`, `TrackState`, `TargetClass`.
- [x] **Stable IDs maintained**: IDs stay attached across consecutive frames (e.g. Swimmer `#04`).
- [x] **Multiple swimmers tracked independently**: Multi-target association validated in unit tests and real footage.
- [x] **Temporary detection gaps bridged**: 3 gaps bridged on sample footage; coasting state verified.
- [x] **Track lifecycle states work**: `NEW`, `TRACKED`, `COASTING`, `LOST` fully operational.
- [x] **Observation history populated**: `observation_history` grows monotonically and chronologically.
- [x] **Tracking visualization works**: `draw_tracks()` renders IDs, boxes, banners, and trajectory breadcrumbs.
- [x] **Unit tests pass**: 15 new tracking tests passing.
- [x] **Existing 46 tests pass**: Total suite at 61 passing tests.
- [x] **Existing detection pipeline still works**: `detect_video.py` verified intact.
- [x] **Existing aerial video produced**: `runs/m4/seadronessee_tracking_sample.mp4` generated.
- [x] **Unseen aerial video produced**: `runs/m4/seadronessee_tracking_unseen.mp4` generated.
- [x] **Results visually inspected**: Verified identity permanence and trajectory overlays.
- [x] **Performance metrics recorded**: 0.22 ms tracking latency, 22.03 FPS pipeline throughput.
- [x] **M4 report written**: `docs/m4_bytetrack_report.md`.
