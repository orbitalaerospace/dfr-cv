# Configuration Templates (`configs/`)

This directory contains configuration templates for dataset definitions and training hyperparameters for the maritime drone computer vision pipeline.

---

## Files

1. `dataset_maritime_template.yaml`:
   - Standard Ultralytics YOLO dataset configuration template mapped to the canonical `TargetClass` domain ontology.
   - Defines sequence-independent paths for `train`, `val`, and `test` splits.

2. `train_maritime_template.yaml`:
   - Hyperparameter reference specification designed for high-altitude and multi-scale aerial maritime perception.
   - Configures mosaic, scale jittering, and anchor optimization for small-object ($10\text{--}80\text{ px}$) detection.

