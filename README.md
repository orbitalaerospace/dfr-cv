# Autonomous Aerial Incident Perception System (`dfr-cv`)

`dfr-cv` is an aerial computer vision system designed for autonomous first responder and search-and-rescue (SAR) drones. The system continuously observes the environment, detects and tracks subjects of interest, performs temporal behavior and persistence reasoning, and emits structured incident alerts to autonomy and command-and-control layers.

> **System Scope & Boundary:**  
> `dfr-cv` is strictly a **perception system**. It does **not** directly control drone flight dynamics, motor ESCs, PID stabilization loops, or low-level flight actuation. Downstream autonomy systems consume the structured incident alerts emitted by this pipeline.

---

## Target Architecture

The production pipeline follows a decoupled, feed-forward perception architecture:

```text
Camera / Video Stream
        │
        ▼
  Preprocessing (Letterboxing, Tensor Normalization)
        │
        ▼
 Spatial Detector (High-Altitude Small Target Localization)
        │
        ▼
Multi-Object Tracker (Motion-Compensated ByteTrack / BoT-SORT)
        │
        ▼
Tracklet & Temporal Reasoning (Sliding Window Kinematics & Persistence)
        │
        ▼
 Incident State Machine (Dual-Threshold Confidence Hysteresis)
        │
        ▼
 Structured Incident JSON (Immutable Validated Payload)
        │
        ▼
Autonomy / Mission Planner / Alert Bus
```

The first operational vertical slice focuses on **aerial water-surface perception** (swimmer and floater detection, persistent tracking across waves and sun glint, and potential drowning/distress identification) grounded on the **SeaDronesSee** benchmark dataset.

---

## Project Status: Milestone 1 Complete

Development proceeds through gated, test-driven engineering milestones.

### Milestone 1 — Foundation, Domain Models, and Data Contracts

- **Status:** **COMPLETE**
- **Objective:** Establish the clean, strongly typed internal language and data contracts of the system before introducing machine learning models or tracking algorithms.

#### What is Implemented in Milestone 1:
1. **Canonical Domain Enums (`src/domain/enums.py`):**
   - `TargetClass`: `PERSON_SURFACE`, `SWIMMER`, `FLOATER`, `LIFE_JACKET`, `WATERCRAFT`, `UNKNOWN`.
   - `TrackState`: `NEW`, `TRACKED`, `COASTING`, `LOST`.
   - `IncidentStatus`: `DETECTED`, `CANDIDATE`, `CONFIRMED`, `RESOLVED`, `REJECTED`.
   - `AlertSeverity`: `INFO`, `LOW`, `MEDIUM`, `HIGH`, `CRITICAL`.
   - *Design rule:* 100% decoupled from ML frameworks and dataset-specific numeric class IDs.
2. **Strict Pydantic Schemas (`src/schemas/`):**
   - `BoundingBox`: Immutable pixel-space box enforcing $x_1 \ge 0$, $y_1 \ge 0$, $x_2 > x_1$, $y_2 > y_1$.
   - `Detection`: Single-frame spatial detection with bounded confidence ($0.0 \le c \le 1.0$) and frame counters.
   - `Tracklet`: Multi-frame object identity tracking contract with chronological timestamp validation.
   - `UavTelemetry`: Optional drone attitude, altitude AGL, and GPS fix contract with physical range validation.
   - `IncidentAlert`: Structured, serializable perception incident alert payload.
3. **Serialization Utilities (`src/schemas/serialization.py`):**
   - Type-safe, validated JSON/dict round-trip helpers with strict error handling.
4. **Environment & Hardware Verification (`tools/verify_env.py`):**
   - Cross-platform hardware diagnostics detecting Apple Silicon MPS, NVIDIA CUDA, and CPU fallback.
5. **Clean Packaging (`pyproject.toml`):**
   - Modern PEP 621 package metadata with separated core, ML, and test dependencies.
6. **Unit Test Suite (`tests/unit/`):**
   - 26 comprehensive unit tests validating bounds, immutability, schema rejection, and serialization fidelity.

#### What is Intentionally NOT Implemented Yet:
- Neural network weights and model architectures (Milestone 2).
- Spatial detection inference loops (Milestone 2).
- ByteTrack / BoT-SORT multi-object tracking logic (Milestone 3).
- Temporal behavior reasoning and incident state machines (Milestone 4).
- End-to-end video streaming pipelines and visual annotators (Milestone 5).
- MAVLink, ROS2, or low-level flight control interfaces.

*(Note: Legacy prototype scripts in `src/use_cases/` are preserved for historical reference and will be systematically deprecated in subsequent milestones).*

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

### 2. Verify Hardware Acceleration

Run the diagnostic utility to verify your compute backend (Apple Silicon MPS, NVIDIA CUDA, or CPU fallback):

```bash
python tools/verify_env.py
```

Expected output on Apple Silicon (M3):
```text
============================================================
 Aerial Perception System — Environment Verification
============================================================

[Host Platform]
  OS / Kernel   : macOS-...-arm64
  Python Version: 3.13.x

[PyTorch & Acceleration Backends]
  PyTorch Version: 2.14.0
  Apple Silicon MPS Available: True (Built: True)
  NVIDIA CUDA Available      : False (Devices: 0)

[Device Selection]
  Active Compute Target: mps [Apple Silicon GPU Accelerated]
============================================================
```

### 3. Running Unit Tests

Execute the Milestone 1 unit test suite:

```bash
python -m pytest tests/unit -v
```

Execute the full test suite (including legacy regression tests):

```bash
python -m pytest -v
```

---

## Roadmap

| Milestone | Scope | Status |
| :---: | :--- | :---: |
| **M1** | **Foundation, Domain Models & Data Contracts** | **COMPLETED** |
| **M2** | Baseline Spatial Detector & SeaDronesSee Training | *Pending* |
| **M3** | Motion-Compensated Multi-Object Tracking Engine | *Pending* |
| **M4** | Tracklet Temporal Reasoning & Incident State Machine | *Pending* |
| **M5** | End-to-End Prerecorded Video Pipeline & 15-Point Test Matrix | *Pending* |
