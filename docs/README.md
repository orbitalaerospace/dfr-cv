# DFR-CV Documentation Index

This directory contains architectural specifications, experimental forensics, evaluation benchmarks, and engineering milestone reports for the `dfr-cv` autonomous search-and-rescue aerial computer vision system.

---

## Documentation Categories

```text
docs/
├── architecture/      # System architecture, schemas, and domain ontologies
├── dataset/           # Maritime dataset engineering specifications & fine-tuning guides
├── evaluation/        # Quantitative evaluation, ground-truth benchmarks, and forensic audits
├── experiments/       # Milestone engineering reports (M2 through M5) & root-cause audits
├── deployment/        # Edge AI hardware deployment (NVIDIA Jetson Orin Nano) specifications
└── development/       # Historical baseline audits & engineering restart documentation
```

---

### 1. Architecture (`docs/architecture/`)
- [`m2_1_ontology_correction.md`](file:///Users/pawaneswaran/Desktop/Work/PROJECTS/drone-cv/dfr-cv/docs/architecture/m2_1_ontology_correction.md): Canonical domain ontology definition, separation of `TargetClass.PERSON` vs `SWIMMER` / `WATERCRAFT`, and framework-agnostic Pydantic V2 data contracts.

### 2. Dataset Engineering (`docs/dataset/`)
- [`FINE_TUNING.md`](file:///Users/pawaneswaran/Desktop/Work/PROJECTS/drone-cv/dfr-cv/docs/dataset/FINE_TUNING.md): YOLO model fine-tuning workflow, dataset layout requirements, and edge export conventions.
- See also the comprehensive [Dataset Engineering Specification](file:///Users/pawaneswaran/Desktop/Work/PROJECTS/drone-cv/dfr-cv/data/README.md) in `data/README.md`.

### 3. Evaluation & Benchmarks (`docs/evaluation/`)
- [`swimmer_ground_truth_evaluation.md`](file:///Users/pawaneswaran/Desktop/Work/PROJECTS/drone-cv/dfr-cv/docs/evaluation/swimmer_ground_truth_evaluation.md): Quantitative spatial bounding-box evaluation measuring True Positives, False Positives, AP50, and precision-recall across controlled inference configurations (`R0`, `R2`, `R3`, `R4`) on moving approaches and synthetic video.
- [`m3_drowning_model_audit.md`](file:///Users/pawaneswaran/Desktop/Work/PROJECTS/drone-cv/dfr-cv/docs/evaluation/m3_drowning_model_audit.md): Forensic model evaluation investigating spatial swimmer detection limits on drone footage.

### 4. Milestone Experiments & Forensics (`docs/experiments/`)
- [`detector_root_cause_audit_v2.md`](file:///Users/pawaneswaran/Desktop/Work/PROJECTS/drone-cv/dfr-cv/docs/experiments/detector_root_cause_audit_v2.md): Forensic root-cause analysis demonstrating the scale mismatch between SeaDronesSee high-altitude training priors and close-range aerial swimmers, proving semantic boat hull confusion.
- [`m2_inference_report.md`](file:///Users/pawaneswaran/Desktop/Work/PROJECTS/drone-cv/dfr-cv/docs/experiments/m2_inference_report.md): Milestone 2 baseline YOLO adapter and pipeline validation.
- [`m3_seadronessee_detector_report.md`](file:///Users/pawaneswaran/Desktop/Work/PROJECTS/drone-cv/dfr-cv/docs/experiments/m3_seadronessee_detector_report.md): Milestone 3 SeaDronesSee model integration, class taxonomy mapping, and maritime detection.
- [`m4_bytetrack_report.md`](file:///Users/pawaneswaran/Desktop/Work/PROJECTS/drone-cv/dfr-cv/docs/experiments/m4_bytetrack_report.md): Milestone 4 ByteTrack multi-object tracking implementation, 2-stage IoU association, and Kalman filter tuning.
- [`m4_5_track_diagnostics_report.md`](file:///Users/pawaneswaran/Desktop/Work/PROJECTS/drone-cv/dfr-cv/docs/experiments/m4_5_track_diagnostics_report.md): Milestone 4.5 temporal feature extraction and track kinematics diagnostics.
- [`m5_temporal_distress_report.md`](file:///Users/pawaneswaran/Desktop/Work/PROJECTS/drone-cv/dfr-cv/docs/experiments/m5_temporal_distress_report.md): Milestone 5 temporal behavior classifier, multi-feature distress scoring, and 4-state hysteresis state machine.

### 5. Deployment (`docs/deployment/`)
- Contains deployment guidelines, Jetson Orin Nano hardware specifications, and TensorRT optimization requirements.

### 6. Development History (`docs/development/`)
- [`mvp_restart_audit.md`](file:///Users/pawaneswaran/Desktop/Work/PROJECTS/drone-cv/dfr-cv/docs/development/mvp_restart_audit.md): Project-state audit (October 10, 2026) establishing baseline architecture and identifying MVP integration targets.
- Root Technical Audit: [DFR_CV_COMPREHENSIVE_AUDIT.md](file:///Users/pawaneswaran/Desktop/Work/PROJECTS/drone-cv/dfr-cv/DFR_CV_COMPREHENSIVE_AUDIT.md).

