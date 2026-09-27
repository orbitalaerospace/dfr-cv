# M2.1 Ontology Correction Report

## Problem
In Milestone 2, the Ultralytics detector adapter (`DEFAULT_COCO_MAPPING`) was configured as:
```python
DEFAULT_COCO_MAPPING = {
    0: TargetClass.PERSON_SURFACE,
    8: TargetClass.WATERCRAFT,
}
```
This was semantically incorrect. 

A generic COCO `0: "person"` detection indicates only that a generic object detector recognized a human body. It contains zero domain evidence that the subject is in water, at a water surface, or in distress. Renaming a generic person detection to `PERSON_SURFACE` introduced ungrounded semantic assumptions into the perception pipeline. 

The adapter must faithfully reflect the detector's native prediction without hallucinating operational context.

## Correction
The canonical domain enum and the default COCO adapter mapping were updated to introduce a generic `PERSON` class:

```python
# src/domain/enums.py
class TargetClass(str, Enum):
    PERSON = "person"                  # Generic human entity
    PERSON_SURFACE = "person_surface"  # Water-surface human entity (for future water models)
    SWIMMER = "swimmer"                # Active swimmer entity
    FLOATER = "floater"                # Passive floating entity
    LIFE_JACKET = "life_jacket"        # Life-saving flotation appliance
    WATERCRAFT = "watercraft"          # Marine vessel / boat / jetski
    UNKNOWN = "unknown"
```

The adapter mapping now strictly preserves model knowledge:
```python
# src/detectors/ultralytics_adapter.py
DEFAULT_COCO_MAPPING: Dict[int, TargetClass] = {
    0: TargetClass.PERSON,       # COCO 0 "person" -> generic TargetClass.PERSON
    8: TargetClass.WATERCRAFT,   # COCO 8 "boat"   -> TargetClass.WATERCRAFT
}
```

## Raw Model Evidence
A direct inspection of raw tensor outputs on Frame 0 of `water_person_test.mov` confirmed the exact native model prediction:
- **Raw Class ID:** `0`
- **Raw Class Name:** `"person"`
- **Observed Confidence:** `0.658`
- **Raw Bounding Box (xyxy):** `[1181.5, 1217.6, 1708.8, 1552.0]`
- **Mapped Domain Target:** `TargetClass.PERSON` (`"person"`)

## Real Video Test
The external test video was re-run using the exact same weights, confidence threshold, and compute device:

- **Input Video:** `data/samples/water_person_test.mov` (Resolutions: $3420 \times 2214$ @ 32.8 FPS)
- **Model Used:** `models/yolo11n.pt` (generic COCO weights)
- **Compute Device:** Apple Silicon GPU (`mps`)
- **Frames Processed:** 76 frames
- **Total Execution Wall Time:** 6.13 s
- **Pipeline Throughput:** **12.40 FPS**
- **Mean Frame Latency:** 48.25 ms
- **Frames with Detections:** 71 / 76 (93.4%)
- **Total Detections Emitted:** 83
- **Class Breakdown:**
  - `person`: 76 detections (confidence range: 0.219 – 0.887)
  - `watercraft`: 7 detections
- **Output Artifact:** `runs/m2/external_water_test_corrected.mp4`

## Before vs After

### Before Correction (M2):
- Native Class: `person` $\longrightarrow$ Mapped Label: `person_surface`
- Displayed Output: `person_surface 0.65`
- *Flaw:* Falsely implied water-surface context from a generic indoor/outdoor COCO model.

### After Correction (M2.1):
- Native Class: `person` $\longrightarrow$ Mapped Label: `person`
- Displayed Output: `person 0.65`
- *Integrity:* Semantically honest representation of the detector's actual prediction.

The model itself was **not** changed; only the semantic translation was made honest.

## Result
The corrected output is now **100% semantically faithful** to the model's native predictions. The system no longer claims domain-specific water capability when running generic COCO weights.

## Tests
Full regression test suite was executed:
- **Total Tests:** 42 passed in 3.55s
  - 26 Milestone 1 schema & serialization tests: `PASSED`
  - 8 Milestone 2 adapter & ontology tests (including explicit checks that COCO 0 maps to `PERSON` and NOT `PERSON_SURFACE`): `PASSED`
  - 8 Legacy prototype tests: `PASSED`

## Next Step
Next step is to identify and integrate a genuinely domain-specific aerial/water detector.
