# M2 Real CV Integration Report

## Model
- **Model Architecture:** YOLO11-Nano (`yolo11n.pt`)
- **Source:** Local pretrained weights (`models/yolo11n.pt`)
- **Weights:** 5.4 MB PyTorch checkpoint (COCO 80 classes)
- **Role / Classification:** Explicitly designated as `generic detector / pipeline validation` (NOT a dedicated drowning detector)
- **Mapped Classes:**
  - Class 0 (`person`) $\rightarrow$ `TargetClass.PERSON_SURFACE`
  - Class 8 (`boat`) $\rightarrow$ `TargetClass.WATERCRAFT`
- **Model Input Size:** Dynamic / $640 \times 640$ letterboxed

## Input Data
- **Dataset / Source:** Official SeaDronesSee maritime aerial footage (Lake Constance drone flight sequence)
- **Image Artifact:** `data/samples/aerial_water_frame0.jpg` ($640 \times 360$ RGB)
- **Video Artifact:** `data/samples/aerial_water_sample.mp4` (41 frames, $640 \times 360$ @ 10.0 FPS)
- **Relevant Targets:** Small boats, wakeboarders, and persons in water observed from an oblique UAV altitude

## Hardware
- **Machine:** Apple Mac (Apple M3, ARM64)
- **OS / Kernel:** macOS Sonoma / Darwin 26.6.2
- **Compute Device:** Apple Silicon MPS (`mps` backend via PyTorch 2.14.0)

## Results

### 1. Still Aerial Image Inference (`aerial_water_frame0.jpg`)
- **Input Resolution:** $640 \times 360$
- **Total Pipeline Latency:** 2105.65 ms (cold start / Metal shader compile included)
- **Model Pure Inference Latency:** 8.59 ms
- **Preprocessing Latency:** 623.03 ms
- **Postprocessing Latency:** 12.92 ms
- **Detections Produced:** 4 targets
- **Observed Detections:**
  1. `watercraft`: conf=0.532, box=[578.8, 97.3, 603.9, 113.9], area=416 px
  2. `watercraft`: conf=0.304, box=[476.9, 80.3, 503.8, 90.4], area=270 px
  3. `watercraft`: conf=0.265, box=[477.5, 69.6, 557.0, 91.9], area=1779 px
  4. `watercraft`: conf=0.256, box=[628.0, 70.8, 639.8, 82.6], area=139 px

### 2. Video Stream Inference (`aerial_water_sample.mp4`)
- **Total Frames Processed:** 41 frames
- **Total Elapsed Wall Time:** 2.89 seconds
- **Effective Pipeline Throughput:** **14.17 FPS**
- **Mean Frame Latency:** 66.97 ms (includes decoding, inference, schema validation, and rendering)
- **Frames with Detections:** 41 / 41 (100.0% coverage)
- **Total Detections Emitted:** 127
- **Observed Confidence Range:** 0.201 – 0.674 (Mean Confidence: 0.450)
- **Class Breakdown:**
  - `watercraft`: 127 detections across 41 frames

## Visual Result
- **Annotated Image Output:** `runs/m2/annotated.jpg`
- **Annotated Video Output:** `runs/m2/annotated.mp4`

Both artifacts visibly show:
- Precision bounding boxes outline active boats and watercraft across water waves.
- High-contrast banner labels with target class (`watercraft`) and confidence score (e.g., `0.53`).
- Real-time HUD status bar indicating frame ID, detection count, latency in milliseconds, and active compute device (`mps`).

## Failure Modes & Honest Evaluation
1. **Missed Small Swimmers (Domain Mismatch):**
   The generic COCO-trained detector reliably tracked moving watercraft ($>15 \times 15$ px), but failed to emit high-confidence detections on small swimmers ($<10 \times 10$ px) in water. In COCO, human priors assume standing/walking figures with visible torsos and limbs, whereas aerial water targets are often heads bobbing in waves.
2. **Wave Wake False Positives:**
   White water turbulence and froth trailing motorboat wakes generated occasional transient false boat detections at low confidence ($0.20 - 0.25$).
3. **Scale Sensitivity:**
   Without sliced-window inference (SAHI) or high-resolution tiling ($1280+$ px), standard 640-scale downsampling attenuates sub-16 pixel features.

## Conclusion
- **What Worked:**
  The end-to-end integration spike succeeded: real aerial drone footage was fed through a real neural network on Apple Silicon MPS, parsed through an abstract `BaseDetector` adapter, transformed strictly into immutable M1 `Detection` schemas, rendered with crisp visual overlays, and timed with actual hardware metrics.
- **What Did Not:**
  Generic COCO weights are inadequate for small-target swimmer search and rescue.
- **Baseline Suitability:**
  This model serves as an effective **generic pipeline validation baseline** proving architectural integrity, but cannot be deployed as the operational SAR perception model.

## Next Step
The single most important next engineering action is **fine-tuning a small-target YOLO architecture on the SeaDronesSee-OD v2 dataset** (`swimmer`, `floater`, `life_jacket`), followed by integrating a multi-object tracker (ByteTrack) to maintain track IDs across wave occlusions.
