# Maritime Dataset Engineering Specification (`data/`)

This document defines the engineering standards, provenance protocols, annotation conventions, and quality control procedures for all search-and-rescue (SAR) aerial maritime datasets in `dfr-cv`.

---

## 1. Directory Structure

```text
data/
├── README.md               # This specification
├── raw/                    # Immutable original video footage and sensor logs (read-only)
├── interim/                # De-duplicated, sliced, and standardized frame sequences
├── processed/              # Normalized, annotated, sequence-split YOLO/COCO datasets
├── metadata/               # Flight manifests, camera calibration, and GPS/attitude logs
├── reports/                # Dataset quality audits, class distributions, and IoU reports
├── samples/                # Existing historical development sample clips (preserved)
└── external_test/          # Existing external test footage (preserved)
```

---

## 2. Raw Data Preservation & Immutability

1. **Write-Once Principle:** Files placed in `data/raw/` are **strictly immutable**. Never modify, rename, re-encode, or overwrite raw video files or sensor telemetry.
2. **Cryptographic Checksums:** Every raw video file must have a companion `.sha256` checksum recorded at ingestion time.
3. **Format Integrity:** Retain original camera containers (`.mp4`, `.mov`, `.mkv`), native resolution ($4\text{K}$, $1080\text{p}$), native frame rates ($24\text{--}60\text{ FPS}$), and embedded metadata tracks (DJI subtitle streams, EXIF, or telemetry tracks).
4. **Storage Tiering:** Large raw videos and tar archives are excluded from Git via `.gitignore`. Large binaries must be managed via Cloud Storage / DVC remotes.

---

## 3. Dataset Provenance & Source Manifests

Every ingest sequence must be documented in a structured JSONL manifest in `data/metadata/manifest.jsonl`.

> [!NOTE]
> **Fictional Schema Template:** The following manifest entry is a structural schema illustration for specification purposes. It does not represent actual recorded flights or real ingested video files.

```json
{
  "_schema_version": "1.0.0",
  "_comment": "FICTIONAL EXAMPLE TEMPLATE ONLY - NOT ACTUAL FLIGHT FOOTAGE",
  "sequence_id": "SAR-FLIGHT-YYYY-MM-DD-LOCATION-01",
  "source_file": "raw/EXAMPLE_SOURCE_VIDEO.MP4",
  "sha256": "0000000000000000000000000000000000000000000000000000000000000000_EXAMPLE_ONLY",
  "drone_platform": "Example-UAV-Platform (e.g., DJI Matrice 300 RTK)",
  "sensor_payload": "Example-Sensor-Payload (e.g., Zenmuse H20T)",
  "environment": "Fictional Ocean / Lake Location",
  "water_condition": "Beaufort scale 2-3 (Example)",
  "lighting": "Daylight / Overcast (Example)",
  "altitude_range_agl_m": [15.0, 60.0],
  "gimbal_pitch_deg": -45.0,
  "native_resolution": [3840, 2160],
  "native_fps": 30.0,
  "total_frames": 1800,
  "target_entities": ["example_targets"],
  "attribution_and_license": "Example License / Data Source"
}
```

---

## 4. Annotation Conventions & Ontology Mapping

All bounding box annotations must conform to the canonical domain ontology defined in [`src/domain/enums.py`](file:///Users/pawaneswaran/Desktop/Work/PROJECTS/drone-cv/dfr-cv/src/domain/enums.py):

| Canonical Class (`TargetClass`) | YOLO Class ID | Visual Boundary Criteria & Edge Rules |
|---|:---:|---|
| **`SWIMMER`** | `0` | Any actively swimming human. Bounding box must enclose the **head, torso, and active stroke splash envelope**. Do not crop tightly to head only. If limbs are kicking, enclose the contiguous water disturbance caused by stroke locomotion. |
| **`FLOATER`** | `1` | A human passively floating on the surface without active stroke progression (treading water, back floating, or stationary). |
| **`LIFE_JACKET`** | `2` | A worn personal flotation device (PFD), life vest, or wearable collar. |
| **`LIFE_SAVING_APPLIANCE`** | `3` | Floating rescue rings, lifebuoys, inflatable life rafts, or rescue tubes separate from or supporting a person. |
| **`WATERCRAFT`** | `4` | Motorized boats, rescue RIBs, kayaks, jet skis, and paddleboards. |
| **`BUOY`** | `5` | Navigational markers, channel markers, mooring buoys, and fishing floats. |

### Architectural Analysis: Detection Classes vs. Behavioral Responsibilities

An essential architectural question is whether **`SWIMMER`** and **`FLOATER`** should be separate static spatial detection classes or unified into a single detection class (`PERSON_SURFACE` or `SWIMMER`), leaving movement state to temporal behavior classification:

1. **Static Visual Ambiguity:** In a single frame taken from a high-altitude drone ($30\text{--}80\text{m}$ AGL), an active swimmer between stroke cycles is visually indistinguishable from a passive floater. Forcing human annotators or single-frame detectors to guess active vs. passive locomotion from static pixels introduces severe annotation noise.
2. **Track Fragmentation in Multi-Object Tracking:** The tracker ([`src/tracking/byte_tracker.py`](file:///Users/pawaneswaran/Desktop/Work/PROJECTS/drone-cv/dfr-cv/src/tracking/byte_tracker.py#L135)) enforces class matching by default (`enforce_class_match=True`). If a person transitions from active freestyle stroke (detected as `swimmer`) to brief treading/floating (detected as `floater`), the Hungarian association penalizes the cross-class match with infinite cost ($1.0$). This immediately **terminates the active track and spawns a new track identity**, causing catastrophic identity switches and fragmented kinematic histories.
3. **Clean Architectural Separation of Concerns:**
   - **Tier 2 (Spatial Detection):** Answers *WHAT* and *WHERE* on a single frame (e.g. human entity on water surface: `PERSON_SURFACE` or `SWIMMER`).
   - **Tier 3 (Multi-Object Tracking):** Maintains temporal identity across occlusions and waves regardless of instantaneous body posture.
   - **Tier 4 (Temporal Behavior Analysis):** Answers *WHAT ARE THEY DOING* over a rolling observation window (e.g. `NORMAL` directed swimming vs. `FLOATING` passive drift vs. `DISTRESS_CONFIRMED` struggle).

*Provisional Recommendation:* For future dataset annotation and model training, human targets on water should be annotated as a single primary class (`swimmer` or `person_surface`), delegating active vs. passive vs. distressed status to downstream temporal kinematics.

### Minimum Spatial Visibility Rule *(Provisional Recommendation)*
- As an initial baseline recommendation pending empirical validation on small targets, any target smaller than **$8 \times 8\text{ px}$** in input resolution with $<0.30$ visual contrast against surrounding water should be flagged with an `occluded` or `uncertain` attribute rather than forced bounding.
- When a swimmer temporarily submerges beneath a wave crest, **do not annotate imaginary bounding boxes under water**. Allow the tracker coasting logic to bridge the gap.

---

## 5. Quality Control & Validation Procedures

1. **Dual-Pass Annotation & Inter-Annotator Agreement *(Provisional Recommendation)*:**  
   High-value evaluation clips should be labeled by two independent reviewers. An initial recommended agreement target is bounding box **$\text{IoU} \ge 0.85$** on identical frames, subject to empirical validation across small ($<32\text{ px}$) targets where single-pixel edge variations significantly shift IoU.
2. **Boundary Sanity Verification:**  
   Automated validation scripts must verify:
   - $x_1 \ge 0$, $y_1 \ge 0$, $x_2 > x_1$, $y_2 > y_1$.
   - Normalized coordinates strictly in $[0.0, 1.0]$.
   - Zero degenerate boxes ($\text{width} = 0$ or $\text{height} = 0$).
3. **Hard Negative Verification *(Provisional Recommendation)*:**  
   Every dataset version should provisionally incorporate approximately **$20\%$ background negative images** (choppy water, sun glint, breaking surf, empty buoys, debris) with **zero annotation boxes** to evaluate and suppress false positive detector alarms. The exact optimal proportion must be tuned empirically based on measured false alarm rates.

---

## 6. Duplicate & Near-Duplicate Detection

Drones frequently hover in place, producing hundreds of nearly identical frames:
1. **Perceptual Hashing (pHash):** Frame sequences extracted from raw video must be filtered using difference hashing (dHash) or perceptual hashing (pHash).
2. **Hover Decimation *(Provisional Recommendation)*:** For static hovering drone footage, downsampling extraction to a recommended baseline of **$2\text{--}5\text{ FPS}$** can mitigate frame redundancy while capturing victim struggle dynamics. The exact sampling rate remains under evaluation.

---

## 7. Sequence-Level Train / Validation / Test Splitting

> [!CAUTION]
> **NEVER PERFORM RANDOM FRAME-LEVEL SHUFFLING ACROSS SPLITS.**  
> Randomly shuffling video frames into train and test sets causes severe temporal data leakage, artificially inflating mAP metrics while producing catastrophic failure on unseen real flights.

### Splitting Rules:
1. **Sortie / Video Independence:** The train, validation, and test splits must be partitioned strictly by **entire flight sequences or video sorties**.
2. **Environmental Disjointness:** If possible, test splits should evaluate water bodies, altitudes, or lighting conditions completely absent from the training set.
3. **Class Representation:** Ensure every target class (`SWIMMER`, `FLOATER`, `WATERCRAFT`, `BUOY`) appears in both training and test partitions with balanced scale variations.

---

## 8. Dataset Versioning Protocol

1. Datasets must follow semantic versioning: `dfr-maritime-v<MAJOR>.<MINOR>.<PATCH>`.
   - `MAJOR`: Fundamental schema change or new core sensor modality.
   - `MINOR`: Addition of new flight sequences or newly annotated classes.
   - `PATCH`: Bounding box label corrections or filtered corrupt frames.
2. Each released dataset partition must include an immutable `dataset_manifest.json` recording total image counts, bounding box distributions, scale histograms, and SHA-256 hashes of all component files.

