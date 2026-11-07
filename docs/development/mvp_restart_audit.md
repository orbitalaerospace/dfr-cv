# MVP Restart Project-State Audit

**Date:** October 10, 2026  
**Repository:** `dfr-cv` (Autonomous Aerial Incident Perception System)  
**Objective:** Project-state audit to establish the baseline and define the shortest safe path to a demonstrable end-to-end MVP by tomorrow.  
**Mode:** Strictly Read-Only Audit (No code or checkpoint modifications performed).

---

## 1. Executive Summary

The `dfr-cv` project possesses a **fully functional, end-to-end aerial computer-vision pipeline** that connects spatial detection, multi-object tracking, temporal kinematics extraction, and behavioral distress classification into rendered annotated video and structured JSON analytics.

All **98 unit and integration tests are passing**. The baseline architecture is sound, modular, and typed via Pydantic V2 contracts. 

The primary blocker encountered prior to the break—swimmer detection drops and boat false alarms on `swimming2.mp4`—was previously diagnosed via a forensic audit (`docs/detector_root_cause_audit_v2.md`). The cause is an object scale mismatch between the high-altitude SeaDronesSee training distribution and close-up camera footage, not a defect in ByteTrack or the M5 behavior classifier. On genuine aerial drone footage (`data/samples/simulated_distress_aerial.mp4`), the system achieves **93.6% swimmer detection recall** and reliably confirms distress behavior.

A demonstrable end-to-end MVP is already operational via `src/pipeline/track_video.py`.

---

## 2. Git Status and Working-Tree Audit

- **Current Branch:** `working` (up to date with `origin/working`).
- **HEAD Commit:** `c279f2d` — *"feat: add temporal distress behavior classifier and behavior state enumeration"*.
- **Recent Commit History:**
  - `c279f2d`: M5 temporal distress classifier & state machine.
  - `de47aee`: M4 ByteTrack multi-object tracking with Kalman filtering & visualization.
  - `62349ad`: M3 SeaDronesSee detector integration & maritime target classes.
  - `73a3cec`: M2.1 ontology correction & generic `TargetClass.PERSON`.
  - `61b88db`: M2 Ultralytics YOLO adapter & detection pipeline.
- **Working-Tree Modifications (Uncommitted):**
  - `modified: src/detectors/ultralytics_adapter.py`: Added optional `classes: Optional[List[int]]` and `imgsz: Optional[int]` parameters to adapter `__init__` and forward prediction pass.
  - `modified: tests/unit/test_detector_adapter.py`: Added unit test `test_seadronessee_classes_filter`.
  - `modified: .DS_Store`: Operating system artifact.
  - `untracked: docs/detector_root_cause_audit_v2.md`: Forensic audit report analyzing SeaDronesSee scale mismatch.
  - `untracked: src/evaluation/`: Contains `detector_video.py` and `audit_experiments.py` (controlled evaluation tools).
  - `untracked: tests/unit/test_audit_experiments.py` & `tests/unit/test_detector_video.py`: Unit tests for evaluation tools.
- **Integrity Status:** All existing work is preserved. No files were discarded or rewritten.

---

## 3. Project Structure, Dependencies, and Environment

### Python Environment
- **Runtime:** Python 3.13.2 (Anaconda / Miniconda base at `/opt/miniconda3/bin/python3`, symlinked via `.venv/bin/python`).
- **Core Library Versions:**
  - `torch`: 2.14.0 (supports Apple Silicon MPS acceleration)
  - `ultralytics`: 8.4.163
  - `opencv-python`: 5.0.0
  - `pydantic`: 2.13.5
  - `numpy`: 2.5.3
  - `scipy`: Installed (utilized by ByteTrack Kalman filter and Hungarian algorithm)

### Project Architecture
```
dfr-cv/
├── models/                     # Checkpoints (SeaDronesSee YOLOv8n, COCO YOLO11n)
├── data/samples/               # Test videos (aerial drone, distress, normal, 4K)
├── runs/                       # Output artifacts (M4, M4.5, M5, audit benchmarks)
├── src/
│   ├── domain/enums.py         # Canonical domain ontologies (TargetClass, TrackState, BehaviorState)
│   ├── schemas/                # Immutable Pydantic V2 contracts
│   │   ├── detection.py        # BoundingBox, Detection
│   │   ├── tracking.py         # Tracklet
│   │   ├── temporal_features.py# TemporalFeatures, TemporalFeatureConfig
│   │   ├── behavior.py         # BehaviorAssessment, BehaviorEvidence, Config
│   │   ├── incident.py         # IncidentAlert
│   │   ├── telemetry.py        # UavTelemetry
│   │   └── serialization.py    # JSON serialization helpers
│   ├── detectors/              # BaseDetector & UltralyticsDetectorAdapter
│   ├── tracking/               # BaseTracker, ByteTracker, KalmanBoxTracker
│   ├── analysis/               # extract_temporal_features, TemporalDistressClassifier
│   ├── visualization/          # draw_detections, draw_tracks
│   ├── pipeline/               # track_video.py (Unified pipeline CLI), detect_video.py
│   ├── evaluation/             # detector_video.py, audit_experiments.py
│   └── use_cases/              # Prototype stubs (SAR, crowd density, etc.)
└── tests/
    ├── unit/                   # 88 core unit tests (contracts, tracker, classifier, etc.)
    └── test_*.py               # 10 legacy prototype test stubs
```

---

## 4. Existing Detector, Checkpoints, and Supported Classes

### Checkpoints
1. `models/seadronessee-yolov8n.pt`:
   - **Architecture:** YOLOv8n (`ultralytics.nn.tasks.DetectionModel`), 3,011,823 parameters.
   - **Training Origin:** Fine-tuned on official SeaDronesSee v2 UAV maritime dataset (Baltic Sea & Lake Constance).
   - **Native Model Classes:**
     - `0`: `swimmer`
     - `1`: `boat`
     - `2`: `jetski`
     - `3`: `life_saving_appliances`
     - `4`: `buoy`
   - **Canonical Domain Mapping:**
     - `swimmer` $\rightarrow$ `TargetClass.SWIMMER`
     - `boat`, `jetski` $\rightarrow$ `TargetClass.WATERCRAFT`
     - `life_saving_appliances` $\rightarrow$ `TargetClass.LIFE_SAVING_APPLIANCE`
     - `buoy` $\rightarrow$ `TargetClass.BUOY`
2. `models/yolo11n.pt`:
   - Baseline COCO model (maps `person` $\rightarrow$ `TargetClass.PERSON`, `boat` $\rightarrow$ `TargetClass.WATERCRAFT`).

### Inference Engine
- Implemented in `src/detectors/ultralytics_adapter.py` (`UltralyticsDetectorAdapter`).
- Auto-detects model signature (SeaDronesSee vs COCO) and selects appropriate mapping.
- Automatically selects Apple Silicon Metal Performance Shaders (`mps`) when available, falling back to CPU.
- Standard default thresholds: `conf_threshold=0.25`, `iou_threshold=0.45`, `imgsz=640`.

---

## 5. ByteTrack Integration, Tracklet Schemas, and Temporal Features

### ByteTrack Implementation (`src/tracking/byte_tracker.py`)
- Detector-agnostic multi-object tracker conforming to `BaseTracker`.
- Standard two-stage association:
  - Stage 1: Associate high-confidence detections (`conf >= 0.40`) with existing tracks using Kalman state IoU matching (match distance threshold 0.8).
  - Stage 2: Associate unmatched tracks with low-confidence detections (`0.10 <= conf < 0.40`).
- **Class Isolation:** `enforce_class_match=True` strictly prevents identity swaps between semantic classes (e.g. Swimmer vs Watercraft).
- **Coasting:** Emits predicted bounding boxes during detection dropout with decaying confidence up to `max_lost=30` frames.

### Tracklet Schema (`src/schemas/tracking.py`)
- Immutable Pydantic model (`Tracklet`):
  - Encapsulates `track_id`, `state` (`NEW`, `TRACKED`, `COASTING`, `LOST`), `current_detection`, `first_seen_timestamp`, `last_seen_timestamp`, and chronological `observation_history: Tuple[Detection, ...]`.
  - Invariant validator guarantees `last_seen_timestamp >= first_seen_timestamp`.

### Temporal Feature Extraction (`src/analysis/temporal_features.py`)
- Extracts 15+ measurable image-space kinematic features from `Tracklet` histories:
  - Duration, observation count, visibility ratio, gap density, max consecutive gap.
  - Net displacement, cumulative path length, tortuosity/straightness ratio.
  - Mean, max, and median image-space velocity (px/s).
  - Mean image-space acceleration (px/s²).
  - Angular deflection count (direction changes $> 45^\circ$).
  - Bounding box area growth ratio (splashing/submersion indicator).
  - Mean confidence.

---

## 6. M5 Behavior Classifier and Configuration

### Architecture (`src/analysis/behavior_classifier.py`)
- Implemented as `TemporalDistressClassifier`.
- Pure temporal analyzer operating on rolling time windows of `Tracklet` observations.
- Four-state finite state machine:
  $$\text{UNKNOWN} \xrightarrow{\ge 5\text{ obs}} \text{NORMAL} \underset{\ge 3\text{ recovery windows}}{\overset{\ge 3\text{ distress windows}}{\rightleftharpoons}} \text{DISTRESS\_CONFIRMED}$$
  *(With intermediate $\text{DISTRESS\_CANDIDATE}$ state during initial window escalation)*.
- **Target Exemption:** Targets not classified as swimmers (e.g. `WATERCRAFT`) are immediately exempted and marked `NORMAL`.

### Multi-Feature Distress Score Formulation
Computes an explainable score in $[0.0, 1.0]$ based on five weighted indicators:
1. **Locomotion Inefficiency (Weight 0.30):** Penalizes low straightness ($\le 0.35$) with high path length (struggling in place).
2. **Directional Volatility (Weight 0.25):** Deflection rate exceeding $1.5\text{ Hz}$.
3. **Bounding Box Growth (Weight 0.20):** Scale expansion ratio exceeding $1.8\times$ (splashing / surface disturbance).
4. **Submersion Gap Density (Weight 0.15):** Intermittent detection dropout ratio exceeding $15\%$.
5. **Kinetic Volatility (Weight 0.10):** Velocity variance and acceleration spikes relative to median speed.

### Default Tuning Parameters (`BehaviorClassifierConfig`)
- `window_seconds = 2.0`
- `stride_seconds = 0.5`
- `min_observations = 5`
- `distress_threshold = 0.55`
- `confirm_windows = 3` (requires 3 consecutive windows above threshold to confirm distress)
- `recovery_windows = 3` (requires 3 consecutive windows below threshold to de-escalate)

---

## 7. Available Test Videos, Datasets, and Generated Outputs

### Video Inventory (`data/samples/`)
| Video File | Resolution | FPS | Frames | Duration | Content & Suitability |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `simulated_distress_aerial.mp4` | 1366x768 | 24.0 | 141 | 5.88s | **Prime Distress Demo**: Real UAV drone footage of swimmer actively struggling. Swimmer recall: 93.6%, 0 boat confusion. |
| `swimming.mp4` | 1366x768 | 24.0 | 141 | 5.88s | **Prime Normal Demo**: Swimmer performing steady directional swimming strokes. |
| `drowning_video.mp4` | 1366x768 | 24.0 | 141 | 5.88s | Real water footage of swimmer in distress. Recall: 86.5%. |
| `aerial_water_sample.mp4` | 640x360 | 10.0 | 41 | 4.10s | Official SeaDronesSee sample footage with small swimmer targets. |
| `aerial_unseen_sequence.mp4` | 3840x2160 | 10.0 | 20 | 2.00s | 4K SeaDronesSee test sequence. |
| `swimming2.mp4` | 1366x768 | 24.0 | 141 | 5.88s | Close-up synthetic swimmer (~466px). Exhibits scale mismatch. |
| `drone_longshot.mp4` | 1366x768 | 24.0 | 141 | 5.88s | Distant drone aerial shot. |
| `water_person_test.mov` | 3420x2214 | 32.8 | 82 | 2.50s | High-resolution water camera footage. |

### Generated Outputs (`runs/`)
- `runs/m5/simulated_distress_analysis.mp4` & `.json`: Confirmed distress behavior output (Track #1 classified as `DISTRESS_CONFIRMED`, score: 0.725).
- `runs/m5/normal_swimming_analysis.mp4` & `normal_analysis.json`: Confirmed normal behavior output (Track #4 classified as `NORMAL`, score: 0.305).
- `runs/detector_audit_v2/`: Comprehensive scale sweep and forensic taxonomy outputs.

---

## 8. Test Suite Status

Pytest run on full test suite:
```
============================== 98 passed in 5.75s ==============================
```
- **Unit Tests:** 88 passing tests covering domain enums, Pydantic data schemas, serialization, telemetry, detector adapters, ByteTrack tracking, temporal kinematics, behavior classification, and pipeline integration.
- **Prototype Tests:** 10 passing mock tests covering legacy use case modules.
- **Failures / Errors:** **0**.

---

## 9. Pipeline CLI Commands and Entry Points

### Primary End-to-End Pipeline
The primary command to run the complete perception, tracking, and behavior pipeline is:

```bash
PYTHONPATH=. .venv/bin/python src/pipeline/track_video.py \
  --source <input_video_path> \
  --output <output_annotated_mp4_path> \
  --export-json <output_analytics_json_path>
```

#### Verified Execution Benchmark (tested during audit on `simulated_distress_aerial.mp4`):
- Compute Device: Apple Silicon MPS
- Pipeline Steps: Frame Capture $\rightarrow$ YOLOv8n Detection $\rightarrow$ ByteTrack Association $\rightarrow$ Kinematic Extraction $\rightarrow$ M5 Behavior Classification $\rightarrow$ Visual Overlay Rendering.
- Mean Frame Latency: ~480 ms (YOLO inference: ~479 ms, ByteTrack: 0.29 ms, Behavior: $<0.1$ ms).
- Visualization output renders:
  - Swimmer bounding boxes.
  - Track ID (`SWIMMER #01`).
  - Historical motion trajectory trail.
  - Real-time behavior banner (`BEHAVIOR: DISTRESS (0.73)` in vivid red, or `BEHAVIOR: NORMAL (0.30)` in green).
  - Upper status bar with FPS, track counts, and latency.

### Auxiliary CLI Utilities
- `src/pipeline/detect_video.py`: Detection-only video annotation.
- `src/pipeline/detect_image.py`: Single-frame detection annotation.
- `src/evaluation/detector_video.py`: Frame-by-frame quantitative detection evaluation.

---

## 10. Implementation Status: Complete vs Incomplete vs Experimental

| Component | Status | Verification & Notes |
| :--- | :---: | :--- |
| **Domain Ontology & Contracts** | **COMPLETE** | Frozen Pydantic V2 schemas for BoundingBox, Detection, Tracklet, TemporalFeatures, BehaviorAssessment, IncidentAlert. |
| **SeaDronesSee Detector Integration** | **COMPLETE** | Ultralytics adapter, auto-mapping, device resolution, thresholding. |
| **ByteTrack MOT Tracker** | **COMPLETE** | Two-stage matching, Kalman filtering, coasting, class isolation. |
| **Temporal Feature Extraction** | **COMPLETE** | 15+ kinematic metrics, sliding window slicing, track diagnostics formatting. |
| **M5 Temporal Behavior Classifier** | **COMPLETE** | 4-state state machine, explainable multi-feature distress score, temporal confirmation windows. |
| **Annotated Video Rendering** | **COMPLETE** | Color-coded status banners, trajectory trails, coasting indicators. |
| **Pipeline CLI Runner** | **COMPLETE** | `src/pipeline/track_video.py` runs end-to-end and outputs MP4 + JSON. |
| **Incident Alert Dispatch (M6)** | **INCOMPLETE** | Contract (`IncidentAlert`) exists in `src/schemas/incident.py`, but `track_video.py` does not currently dispatch alert events to external consumers or log them to an alert stream when `DISTRESS_CONFIRMED` triggers. |
| **Camera Ego-Motion Compensation** | **NOT IMPLEMENTED** | Motion vectors are 2D image-space pixels. Sufficient for hovering/slow drone, but not compensated for rapid panning. |
| **Detector Scale Invariance** | **KNOWN LIMITATION** | SeaDronesSee model is optimized for high-altitude UAV imagery ($10-80\text{ px}$); close-up targets ($>300\text{ px}$) exhibit torso/boat confusion. Multi-scale fine-tuning deferred. |
| **Early Prototype Modules (`src/use_cases/`, `src/track.py`)** | **EXPERIMENTAL** | Legacy scripts from early project ideation. Kept intact to preserve test suite compatibility. |

---

## 11. Prioritized Blockers for Tomorrow's MVP

1. **Blocker 1 (Demo Narrative / Video Selection):**  
   *Risk:* Demonstrating on `swimming2.mp4` will trigger the known detector scale mismatch (boat false positives), distracting from the functioning pipeline.  
   *Mitigation:* Base the MVP demonstration strictly on our verified aerial video assets:
   - **Distress Scenario:** `data/samples/simulated_distress_aerial.mp4` (verified: 93.6% recall, confirmed distress).
   - **Normal Swimming Scenario:** `data/samples/swimming.mp4` or `data/samples/aerial_water_sample.mp4` (verified: steady track, normal behavior).

2. **Blocker 2 (Incident Alert Generation):**  
   *Gap:* The system visually displays distress and writes behavior assessments to JSON, but does not emit a formal `IncidentAlert` object (the M6 contract).  
   *Mitigation:* Wire `IncidentAlert` generation directly into `track_video.py` so that when a track transitions to `DISTRESS_CONFIRMED`, a structured alert is logged and exported.

3. **Blocker 3 (Uncommitted Working-Tree Files):**  
   *State:* Useful adapter parameters (`classes`, `imgsz`) and evaluation modules are uncommitted.  
   *Mitigation:* Stage and commit these cleanly without modifying production defaults.

---

## 12. Shortest Safe Path to a Demonstrable MVP Tomorrow

The shortest, safest path requires **zero architectural rewriting**:

1. **Step 1 (Repository Hygiene):**  
   Commit the existing uncommitted improvements (`ultralytics_adapter.py`, `test_detector_adapter.py`, `src/evaluation/`, and corresponding tests).
2. **Step 2 (Operationalize Incident Alert Emission):**  
   In `src/pipeline/track_video.py`, emit structured `IncidentAlert` objects into a dedicated alerts JSONL output whenever a tracklet is confirmed in distress (`DISTRESS_CONFIRMED`).
3. **Step 3 (Pre-Generate MVP Demo Artifacts):**  
   Run `src/pipeline/track_video.py` on both the normal and distress benchmark videos to generate pre-rendered demonstration MP4s and companion JSON analytics:
   - `demo/normal_swimmer_demo.mp4`
   - `demo/distress_incident_demo.mp4`
   - `demo/incident_alerts.jsonl`
4. **Step 4 (Create a Single Unified Demo Script):**  
   Provide a single CLI command (`scripts/run_demo.sh` or `python -m src.pipeline.track_video ...`) so the user can rerun any sample live during a presentation.

---

**Audit completed.** Code modifications paused pending user review and approval.
