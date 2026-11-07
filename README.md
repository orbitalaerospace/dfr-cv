# Autonomous Aerial Incident Perception System (`dfr-cv`)

`dfr-cv` is an aerial computer vision system designed for autonomous first responder and search-and-rescue (SAR) drones. The system continuously observes maritime environments, detects and tracks human subjects in water across wave crest submersions, extracts temporal kinematic features, evaluates behavioral distress indicators, and emits structured, immutable incident alerts for downstream mission planners and ground stations.

> [!IMPORTANT]
> **System Boundaries & Operational Scope:**  
> 1. `dfr-cv` is strictly a **perception system**. It does **not** directly control drone flight dynamics, motor ESCs, PID stabilization loops, or low-level flight actuation. Downstream autonomy systems consume the structured incident alerts emitted by this pipeline.
> 2. **Behavioral Distress vs Drowning:** The temporal behavior classifier is an **explainable statistical motion heuristic** (locomotion efficiency, directional volatility, bounding box area fluctuation, and submersion gaps). It is **not** a clinically validated medical drowning diagnostic. It does not detect silent or motionless drowning (Instinctive Drowning Response).
> 3. **Operational Maturity:** The system is an active **engineering prototype** undergoing controlled forensic evaluation. It is **not certified for autonomous life-safety operations without human-in-the-loop review**.

---

## System Architecture

The pipeline follows a modular, decoupled feed-forward perception architecture:

```text
Camera / Video Stream (cv2.VideoCapture)
        │
        ▼
Spatial Detector Adapter (UltralyticsDetectorAdapter: SeaDronesSee YOLOv8n / YOLO11n)
        │
        ▼
Canonical Detection Contracts (Pydantic V2 BoundingBox & Detection)
        │
        ▼
Multi-Object Tracker (ByteTracker: 2-Stage Hungarian Matching + 8-State Kalman Filter)
        │
        ▼
Canonical Tracklet Contracts (Pydantic V2 Tracklet & Lifecycle States)
        │
        ▼
Temporal Kinematics & Windowing (extract_temporal_features: 2.0s Window, 0.5s Stride)
        │
        ▼
Behavioral Distress Classifier (TemporalDistressClassifier: 4-State Hysteresis Machine)
        │
        ▼
Incident Lifecycle & Deduplication (IncidentLifecycleTracker: State Transition Gate)
        │
        ▼
Serialized Outputs (draw_tracks Annotated MP4, Diagnostics JSON, Incident Alerts JSONL)
```

---

## Implementation & Engineering Status

| Pipeline Component | Status | Detailed Description |
|---|:---:|---|
| **Domain Ontologies** | **Implemented** | Canonical string enums in [`src/domain/enums.py`](file:///Users/pawaneswaran/Desktop/Work/PROJECTS/drone-cv/dfr-cv/src/domain/enums.py): `TargetClass` (`SWIMMER`, `FLOATER`, `WATERCRAFT`, `BUOY`, etc.), `TrackState`, `BehaviorState`, `IncidentStatus`, `AlertSeverity`. |
| **Data Contracts** | **Implemented** | Strict, immutable Pydantic V2 schemas in [`src/schemas/`](file:///Users/pawaneswaran/Desktop/Work/PROJECTS/drone-cv/dfr-cv/src/schemas): `BoundingBox` (coordinate bounds & non-negativity), `Detection`, `Tracklet`, `TemporalFeatures`, `BehaviorAssessment`, `IncidentAlert`, `UavTelemetry`. |
| **Spatial Detection** | **Implemented** | [`UltralyticsDetectorAdapter`](file:///Users/pawaneswaran/Desktop/Work/PROJECTS/drone-cv/dfr-cv/src/detectors/ultralytics_adapter.py) supporting fine-tuned SeaDronesSee YOLOv8n (`models/seadronessee-yolov8n.pt`) and COCO YOLO11n (`models/yolo11n.pt`). Includes coordinate clamping, class filtering, and latency profiling. |
| **Multi-Object Tracking** | **Implemented** | [`ByteTracker`](file:///Users/pawaneswaran/Desktop/Work/PROJECTS/drone-cv/dfr-cv/src/tracking/byte_tracker.py) with 2-stage Hungarian matching and an 8-state constant velocity Kalman filter in [`src/tracking/kalman.py`](file:///Users/pawaneswaran/Desktop/Work/PROJECTS/drone-cv/dfr-cv/src/tracking/kalman.py). Manages track confirmation (`min_hits=3`), coasting with confidence decay, and termination (`max_lost=30`). |
| **Temporal Kinematics** | **Implemented** | Pure function [`extract_temporal_features`](file:///Users/pawaneswaran/Desktop/Work/PROJECTS/drone-cv/dfr-cv/src/analysis/temporal_features.py) computing straightness ratio, angular deflections per second, bbox growth ratio, detection gap density, and speed volatility. |
| **Behavior Assessment** | **Implemented** | [`TemporalDistressClassifier`](file:///Users/pawaneswaran/Desktop/Work/PROJECTS/drone-cv/dfr-cv/src/analysis/behavior_classifier.py) combining weighted motion features with a 4-state temporal state machine (`UNKNOWN`, `NORMAL`, `DISTRESS_CANDIDATE`, `DISTRESS_CONFIRMED`) and multi-window hysteresis. |
| **Incident Lifecycle** | **Implemented** | `IncidentLifecycleTracker` in [`src/pipeline/track_video.py`](file:///Users/pawaneswaran/Desktop/Work/PROJECTS/drone-cv/dfr-cv/src/pipeline/track_video.py) deduplicating and emitting structured `IncidentAlert` events on confirmed distress and resolution. |
| **Visual Annotator** | **Implemented** | [`draw_tracks`](file:///Users/pawaneswaran/Desktop/Work/PROJECTS/drone-cv/dfr-cv/src/visualization/annotator.py) rendering color-coded bounding boxes, persistent track IDs, trajectory breadcrumbs, and real-time status banners. |
| **Dataset Engineering** | **Incomplete / Pending** | Specifications created in [`data/README.md`](file:///Users/pawaneswaran/Desktop/Work/PROJECTS/drone-cv/dfr-cv/data/README.md). Repository currently contains sample clips (~38s total); lacks standardized, sequence-split training datasets. |
| **Edge Deployment** | **Incomplete / Pending** | Tested on Apple Silicon Mac (`mps`) and CPU fallback. TensorRT FP16/INT8 export and DeepStream integration for NVIDIA Jetson Orin Nano are planned. |

---

## Functionality Classification Matrix

### 1. Implemented & Empirically Evaluated
- **End-to-End Tracking Pipeline ([`src/pipeline/track_video.py`](file:///Users/pawaneswaran/Desktop/Work/PROJECTS/drone-cv/dfr-cv/src/pipeline/track_video.py)):** Evaluated on authentic high-altitude drone footage (`aerial_water_sample.mp4`, `simulated_distress_aerial.mp4`, `drone_longshot.mp4`).
- **Small Aerial Swimmer Detection:** High-altitude small target detection ($10\text{--}80\text{ px}$) achieving $>93\%$ recall on authentic UAV imagery.
- **Occlusion & Submersion Recovery:** ByteTrack Stage 2 association bridging temporary swimmer wave crest submersions.
- **Forensic Scale Diagnosis:** Exhaustive empirical evaluation documenting the scale mismatch between SeaDronesSee priors and close swimmers (see [`docs/experiments/detector_root_cause_audit_v2.md`](file:///Users/pawaneswaran/Desktop/Work/PROJECTS/drone-cv/dfr-cv/docs/experiments/detector_root_cause_audit_v2.md)).

### 2. Unit-Test-Only Functionality
- **Pydantic Validation Invariants:** Boundary validation, negative coordinate rejection, and serialization round-trip fidelity.
- **Kalman Filter Mathematical Propagation:** Covariance symmetry, projection, and update steps on synthetic coordinate inputs.
- **Temporal State Transitions:** Rolling window slicing and state machine transitions on synthetic tracklet sequences.

### 3. Proposed Future Work (Engineering Roadmap)
- **Multi-Scale Maritime Fine-Tuning:** Training on combined high-altitude and medium-altitude aerial swimmer footage with scale augmentations ($0.2\times\text{--}2.0\times$).
- **Camera Motion Compensation (CMC):** Estimating drone ego-motion via affine transforms inside Kalman tracking to prevent drone drift from contaminating kinematic features.
- **Tiled Inference (SAHI):** Sliced high-resolution inference for distant target detection in 4K video feeds.
- **Telemetry Fusion:** Ingesting drone altitude AGL and gimbal pitch from MAVLink / SRT into [`UavTelemetry`](file:///Users/pawaneswaran/Desktop/Work/PROJECTS/drone-cv/dfr-cv/src/schemas/telemetry.py) to calculate metric ground coordinates.
- **TensorRT Optimization:** FP16/INT8 engine export and zero-copy hardware video decoding on NVIDIA Jetson Orin Nano.

---

## Getting Started

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/orbitalaerospace/dfr-cv.git
cd dfr-cv

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Verify Hardware Acceleration & Dependencies

Run the diagnostic utility to verify compute targets (Apple Silicon MPS, NVIDIA CUDA, or CPU fallback) and package readiness:

```bash
python tools/verify_env.py
```

### 3. Running the Test Suite

Execute the core test suite (109 passing tests):

```bash
python -m pytest
```

Execute unit tests only:

```bash
python -m pytest tests/unit -v
```

### 4. Running the Sequential Tracking Pipeline

Process an aerial video stream through detection, ByteTrack tracking, temporal kinematics, and distress classification:

```bash
# Using fine-tuned SeaDronesSee maritime detector
python -m src.pipeline.track_video \
  --source data/samples/simulated_distress_aerial.mp4 \
  --output runs/demo/tracking_output.mp4 \
  --detector seadronessee \
  --conf 0.25 \
  --export-json runs/demo/diagnostics.json \
  --export-alerts runs/demo/alerts.jsonl

# Using general-purpose COCO YOLO11n detector
python -m src.pipeline.track_video \
  --source data/samples/drone_longshot.mp4 \
  --output runs/demo/yolo11n_output.mp4 \
  --detector yolo11n \
  --conf 0.25
```

---

## Documentation Navigation

- **Comprehensive System Audit:** [`DFR_CV_COMPREHENSIVE_AUDIT.md`](file:///Users/pawaneswaran/Desktop/Work/PROJECTS/drone-cv/dfr-cv/DFR_CV_COMPREHENSIVE_AUDIT.md) — Exhaustive technical audit covering architecture, detectors, datasets, tracking, kinematics, edge readiness, and risks.
- **Documentation Index:** [`docs/README.md`](file:///Users/pawaneswaran/Desktop/Work/PROJECTS/drone-cv/dfr-cv/docs/README.md) — Index of all reports organized into `architecture/`, `dataset/`, `evaluation/`, `experiments/`, `deployment/`, and `development/`.
- **Dataset Engineering Specification:** [`data/README.md`](file:///Users/pawaneswaran/Desktop/Work/PROJECTS/drone-cv/dfr-cv/data/README.md) — Specifications for raw preservation, source manifests, annotation standards, and sequence-level train/val/test splits.
- **Experiment Outputs Index:** [`runs/README.md`](file:///Users/pawaneswaran/Desktop/Work/PROJECTS/drone-cv/dfr-cv/runs/README.md) — Catalog of all historical experiment outputs and benchmark JSONs.

---

## Legacy Modules & Deprecation Strategy

The scripts in `src/use_cases/` (`alpr.py`, `fire_smoke_detection.py`, `crowd_density.py`, etc.) and root-level tests `tests/test_*.py` are legacy prototypes predating the maritime search-and-rescue focus. They are preserved for historical reference but are decoupled from the core pipeline. They are slated for archival into an `archive/legacy_prototypes/` directory in a future milestone.
