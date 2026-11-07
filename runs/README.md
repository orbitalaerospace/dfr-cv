# Experiment Outputs and Artifacts Index (`runs/`)

This directory contains historical experiment outputs, quantitative benchmark metrics, rendered tracking videos, and forensic audit results generated across project milestones.

> **CRITICAL DATA PRESERVATION NOTICE:**  
> All files in `runs/` represent permanent empirical evidence from controlled evaluation runs. Do not delete, overwrite, or clean this directory.

---

## Directory Organization & Milestone Index

```text
runs/
├── demo/                   # Demonstrable output videos and annotated frame sequences
├── m2/                     # Milestone 2: Spatial detector adapter verification outputs
├── m3/                     # Milestone 3: SeaDronesSee detector integration benchmarks
├── m4/                     # Milestone 4: ByteTrack multi-object tracking videos and frame dumps
├── m4_5/                   # Milestone 4.5: Track kinematics diagnostics and temporal features JSONs
├── m5/                     # Milestone 5: Temporal distress behavior analysis runs and JSON reports
├── m5_5/                   # Milestone 5.5: Confidence threshold sweep benchmarks on close-range swimmers
├── m6_1/                   # Milestone 6.1: Dual-detector evaluation (SeaDronesSee vs YOLO11n) & alerts JSONL
├── detector_audit_v2/      # Forensic root-cause audit artifacts (scale, resolution, confusion, GT)
└── experiments/            # Controlled configuration sweeps (R0, R1, R2, R3, R4) and comparative evaluations
```

---

### Detailed Summary of Historical Runs

#### 1. Milestone 6.1 — Incident Lifecycle Alerts & Dual-Detector Benchmarks (`runs/m6_1/`)
- **Purpose:** Compares `seadronessee-yolov8n` vs `yolo11n` across test videos (`aerial_water_sample.mp4`, `drone_longshot.mp4`, `vid.mp4`) and records structured `IncidentAlert` JSONL event streams.
- **Key Artifacts:**
  - `seadronessee_drone_longshot_alerts.jsonl`: Confirmed distress alerts dispatched for persistent struggling track.
  - `yolo11n_drone_longshot_alerts.jsonl`: Confirmed distress alerts using generic person detector.
  - `*.json`: Full frame-by-frame tracking and behavior evaluation summaries.
  - `*.mp4`: Rendered videos with bounding boxes, trajectory breadcrumbs, and behavior banners.

#### 2. Forensic Root-Cause Audit (`runs/detector_audit_v2/`)
- **Purpose:** Machine-readable data supporting the forensic root-cause analysis in `docs/experiments/detector_root_cause_audit_v2.md` and `docs/evaluation/swimmer_ground_truth_evaluation.md`.
- **Key Artifacts:**
  - `scale_experiment_results.json`: Empirical scale sweep ($0.25\times\text{--}1.50\times$) proving detection recovery at $0.50\times$.
  - `input_resolution_results.json`: Input resolution sweep ($320\text{--}1280\text{ px}$) demonstrating the resolution conflict.
  - `class_confusion_forensics_results.json`: Quantification of the 163 boat false alarms on human swimmer torso.
  - `swimming2_ground_truth.json`: Algorithmic ground-truth bounding box file for 141 frames of `swimming2.mp4`.
  - `ground_truth_evaluation_results.json`: Formal Precision, Recall, F1, and AP50 metrics.
  - `real_vs_synthetic_results.json`: Domain comparison across genuine UAV, 4K aerial, and synthetic footage.
  - `dataset_gap_analysis_results.json`: Aspect ratio and pixel area disparity factors ($9.8\times$ area mismatch).

#### 3. Milestone 5 & 5.5 — Temporal Behavior & Confidence Sweeps (`runs/m5/`, `runs/m5_5/`)
- **Purpose:** Evaluation of `TemporalDistressClassifier` on normal swimming vs distressed swimming videos.
- **Key Artifacts:**
  - `runs/m5/simulated_distress_analysis.json`: Confirmed distress classification on authentic UAV distress footage.
  - `runs/m5/normal_analysis.json`: Correct `NORMAL` swimming classification on translational swimming footage.
  - `runs/m5_5/swimming2_swimmer_conf*.json`: Sensitivity sweep across confidence thresholds ($0.15, 0.20, 0.25$).

#### 4. Milestone 4 & 4.5 — ByteTrack & Kinematics Diagnostics (`runs/m4/`, `runs/m4_5/`)
- **Purpose:** ByteTrack multi-object tracking stability verification across occlusion and wave submersion.
- **Key Artifacts:**
  - `runs/m4_5/simulated_distress_diagnostics.json`: Extracted kinematics (straightness, direction change rate, area growth).
  - `runs/m4_5/sample_track_diagnostics.json`: Diagnostics on 41-frame Lake Constance SeaDronesSee sample.
  - `runs/m4/seadronessee_tracking_unseen.mp4`: Tracking video on 4K Baltic Sea sequence.

#### 5. Controlled Experiment Sweeps (`runs/experiments/`)
- **Purpose:** Controlled evaluation configurations:
  - `R0`: Baseline (`imgsz=640`, unconstrained classes, `conf=0.25`).
  - `R1`: Native class filtering (`classes=[0]`, `conf=0.25`).
  - `R2`: Class-filtered low-confidence (`classes=[0]`, `conf=0.15`).
  - `R3`: Intermediate resolution (`imgsz=480`, `classes=[0]`, `conf=0.15`).
  - `R4`: Downsampled resolution (`imgsz=320`, `classes=[0]`, `conf=0.15`).

