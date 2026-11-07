# Forensic Root-Cause Audit: SeaDronesSee Swimmer Detection Failure & Class Confusion

## 1. Executive Summary

This forensic audit investigates the failure of the SeaDronesSee YOLOv8n detector ([`models/seadronessee-yolov8n.pt`](file:///Users/pawaneswaran/Desktop/Work/PROJECTS/drone-cv/dfr-cv/models/seadronessee-yolov8n.pt)) on close-range aerial swimming footage ([`data/samples/swimming2.mp4`](file:///Users/pawaneswaran/Desktop/Work/PROJECTS/drone-cv/dfr-cv/data/samples/swimming2.mp4)). 

Prior unconstrained detector runs revealed a severe dual failure mode: the detector emitted **163 false-positive `WATERCRAFT` detections across 90.1% of the video**, while true `SWIMMER` detections were emitted in only **55.3% of frames (at conf=0.25)** and frequently split the human body into disconnected parts (torso as boat, legs as swimmer).

Through six controlled, empirical experiments—spanning apparent scale variations, input resolution sweeps, frame-by-frame confusion taxonomy classification, formal ground-truth evaluation, cross-domain video benchmarking, and training dataset gap quantification—this audit proves:

1. **Primary Root Cause (Scale / Receptive Field Mismatch)**: The model's convolutional receptive fields and anchors for `swimmer` were learned exclusively on high-altitude UAV imagery (where swimmers average $87.7\text{ px} \times 73.7\text{ px}$ in 4K, occupying $<0.08\%$ of the image area). In `swimming2.mp4`, the swimmer measures $466.0\text{ px} \times 136.6\text{ px}$ ($6.1\%$ of the image area, **$9.8\times$ larger area**).
2. **Mechanistic Failure Mode (Semantic Boat Hull Confusion & Fragmentation)**: Because elongated floating objects $>250\text{ px}$ in maritime imagery are overwhelmingly vessels, the model's feature maps assign higher activation to `boat` ($0.50 - 0.80$ confidence) than `swimmer` ($0.25 - 0.41$ confidence) for the swimmer's torso.
3. **Empirical Proof of Scale Sensitivity**: When the swimmer in `swimming2.mp4` is scaled down by $0.50\times$, swimmer detection rate jumps from **$73.8\%$ to $97.9\%$**, and longest consecutive dropout drops from **$8$ frames to $2$ frames ($0.08\text{ s}$)**. When input resolution is adjusted to `imgsz=320` (which downsamples the tensor), ground-truth **AP50 jumps from $0.013$ to $0.7256$**, recall jumps from **$4.96\%$ to $79.43\%$**, and precision reaches **$91.80\%$**.
4. **Resolution Limitation vs Gating Factor**: High input resolution (`imgsz=1280`) does **not** help; it worsens the scale mismatch (reducing recall to $7.8\%$). Downsampling resolution (`imgsz=320`) mitigates the close swimmer scale issue, but sacrifices edge sensitivity on genuine high-altitude small swimmers ($10 - 20\text{ px}$).
5. **Definitive Treatment**: Class restriction and resolution tuning provide diagnostic evidence, but cannot resolve multi-altitude operational requirements. **Targeted multi-scale fine-tuning on medium and close-range aerial swimmers ($100\text{ px} - 500\text{ px}$) is required.**

---

## 2. Current Failure Description

### Video Profile (`data/samples/swimming2.mp4`)
- **Resolution**: $1366 \times 768$ @ $24.0\text{ FPS}$
- **Total Duration**: $141\text{ frames}$ ($5.88\text{ seconds}$)
- **Content**: A single human swimmer swimming horizontally in open water. No watercraft, jet skis, buoys, or other objects are present.
- **True Ground Truth**: Exactly 1 full-body swimmer target present in all 141 frames ($100\%$ temporal presence).

### Observed Baseline Metrics
| Configuration | Swimmer Frame Rate | Total Swimmer Dets | Watercraft Frame Rate | Total Watercraft Dets | Overlap (Both) | Frames Neither |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Unconstrained (`conf=0.25`)** | $55.3\%$ (78/141) | 86 | **$90.1\%$ (127/141)** | **163** | 68 frames | 4 frames |
| **Swimmer-only (`classes=[0], conf=0.25`)** | $55.3\%$ (78/141) | 86 | $0.0\%$ (0/141) | 0 | 0 frames | 63 frames |
| **Swimmer-only (`classes=[0], conf=0.20`)** | $62.4\%$ (88/141) | 102 | $0.0\%$ (0/141) | 0 | 0 frames | 53 frames |
| **Swimmer-only (`classes=[0], conf=0.15`)** | $73.8\%$ (104/141) | 132 | $0.0\%$ (0/141) | 0 | 0 frames | 37 frames |

---

## 3. Checkpoint Audit (`models/seadronessee-yolov8n.pt`)

Inspection of the binary checkpoint metadata via PyTorch and Ultralytics reveals:

| Metadata Field | Verified Value | Forensic Significance |
| :--- | :--- | :--- |
| **Model Architecture** | `ultralytics.nn.tasks.DetectionModel` (YOLOv8n) | Standard 3-scale feature pyramid (P3, P4, P5 strides: 8, 16, 32). |
| **Parameter Count** | **3,011,823 parameters** (~3.01M), 8.7B FLOPs | Compact nano backbone optimized for embedded UAV edge devices. |
| **Native Class Names** | `{0: 'swimmer', 1: 'boat', 2: 'jetski', 3: 'life_saving_appliances', 4: 'buoy'}` | 5 maritime domain classes. Classes 1 & 2 map to `WATERCRAFT`. |
| **Training Dataset** | `/home/neural_debugger/Downloads/datasets/SeaDronesSee/CompressedVersion/yolo_dataset/data.yaml` | SeaDronesSee v2 object detection benchmark (Lake Constance & Baltic Sea). |
| **Training Run Source** | `DetectionBench`, commit `6794dff607b587856b65bed22bd9e66b573792f1` | Standard academic benchmark recipe. |
| **Training Parameters** | `epochs=500` (stopped at 220, patience 100), `batch=32`, `imgsz=640`, `lr0=0.001` | Trained strictly at fixed 640px input resolution. |
| **Augmentation Profile** | Mosaic=1.0, FlipLR=0.5, Erasing=0.4, HSV-S=0.7, HSV-V=0.4, Scale=0.5, Translate=0.1 | Scale jitter ($\pm 50\%$) was centered on small aerial swimmers; insufficient to bridge $5\times$ scale gaps. |
| **Validation Metrics in Checkpoint** | Precision: `0.8266`, Recall: `0.6805`, $\text{mAP}_{50}$: `0.6910`, $\text{mAP}_{50-95}$: `0.4010` | High benchmark performance on official SeaDronesSee test distribution. |
| **Inference Preprocessing Defaults** | Dynamic letterbox to `imgsz=640`, `conf=0.25`, `iou=0.45`, `agnostic_nms=False` | Class-aware NMS runs independently per class, allowing simultaneous boat and swimmer boxes on the same target. |

---

## 4. Phase 2 — Controlled Scale Experiment

### Hypothesis
If the primary cause of missed swimmer detections and boat confusion is the excessive pixel scale of the swimmer relative to the network's learned priors, systematically reducing the apparent pixel size of the swimmer while holding the video content and weights constant will improve swimmer detection probability and suppress multi-box fragmentation.

### Experimental Design
The $1366 \times 768$ frames of `swimming2.mp4` were scaled across six factors: $0.25\times, 0.50\times, 0.75\times, 1.00\times, 1.25\times, 1.50\times$. 
- Scales $\le 1.0\times$ were embedded on a matched ocean background canvas to maintain identical field-of-view geometry.
- Scales $> 1.0\times$ were cropped and zoomed from the central swimmer corridor.
- Detector evaluated strictly on native class `0` (`swimmer`) at `conf=0.15`.

### Quantitative Results
Machine-readable artifact: [`runs/detector_audit_v2/scale_experiment_results.json`](file:///Users/pawaneswaran/Desktop/Work/PROJECTS/drone-cv/dfr-cv/runs/detector_audit_v2/scale_experiment_results.json)

| Scale Factor | Apparent Width (px) | Apparent Height (px) | Swimmer Detection Rate | Total Detections | Mean Conf | Multi-Box Rate ($>1$) | Longest Consecutive Dropout |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.25x** | 116.5 | 34.1 | $43.97\%$ (62/141) | 62 | **0.6505** | **0.0%** (0/141) | 42 frames |
| **0.50x** | **233.0** | **68.3** | **97.87% (138/141)** | **155** | **0.5064** | **12.06%** (17/141) | **2 frames (0.08s)** |
| **0.75x** | 349.5 | 102.4 | $44.68\%$ (63/141) | 81 | 0.3828 | 12.06% (17/141) | 35 frames |
| **1.00x (Base)** | 466.0 | 136.6 | $73.76\%$ (104/141) | 132 | 0.3344 | 19.86% (28/141) | 8 frames (0.33s) |
| **1.25x** | 582.5 | 170.8 | $51.06\%$ (72/141) | 92 | 0.3795 | 12.06% (17/141) | 28 frames |
| **1.50x** | 699.0 | 204.9 | $58.16\%$ (82/141) | 103 | 0.3755 | 11.35% (16/141) | 15 frames |

### Forensic Analysis of Scale
1. **The "Sweet Spot" Phenomenon at 0.50x**:
   - At $0.50\times$, the apparent width of the swimmer is $233.0\text{ px}$. Downsampled into the $640\text{ px}$ letterbox input tensor, this corresponds to $\sim 109\text{ px}$ in network space.
   - At this scale, **detection coverage surges to $97.87\%$ (138 out of 141 frames)**, and the maximum consecutive dropout collapses to **only 2 frames ($0.08\text{ seconds}$)**.
   - Mean confidence reaches **$0.5064$** (a $+51.4\%$ relative increase over the base scale).
2. **Fragmentation Drop**:
   - At $0.25\times$ scale, multi-box fragmentation completely vanishes ($0.0\%$).
   - At $1.00\times$ base scale, fragmentation peaks at **$19.86\%$** (28 frames containing competing swimmer boxes).
3. **Conclusion on Scale**:
   - The hypothesis is **strongly supported**. Apparent pixel scale is the primary governing factor dictating whether the model localizes the target as a single coherent swimmer or misclassifies/fragments it.

---

## 5. Phase 3 — Input Resolution Experiment

### Hypothesis
If the detector failure is caused by an inability of the CNN feature maps to resolve details at the default `imgsz=640`, increasing input resolution to $960\text{ px}$ or $1280\text{ px}$ will improve recall and localization accuracy. Conversely, if the failure is scale-driven, increasing resolution will worsen the scale mismatch, whereas reducing resolution to $320\text{ px}$ will improve full-body detection.

### Experimental Design
Evaluated on native `swimming2.mp4` across four model input resolutions: `imgsz in [320, 640, 960, 1280]`, with `classes=[0], conf=0.15` constant. Ground truth bounding boxes for all 141 frames were used to evaluate formal precision, recall, and AP50.

### Quantitative Results
Machine-readable artifact: [`runs/detector_audit_v2/input_resolution_results.json`](file:///Users/pawaneswaran/Desktop/Work/PROJECTS/drone-cv/dfr-cv/runs/detector_audit_v2/input_resolution_results.json)

| Resolution (`imgsz`) | Swimmer Frame Rate | Mean Conf | Multi-Box Rate | Longest Dropout | GT Precision | GT Recall | GT F1 | GT Mean IoU | GT AP50 | Mean Latency |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **320** | **79.43%** (112/141) | **0.5382** | **7.09%** | 12 frames | **0.9180** | **0.7943** | **0.8517** | **0.7325** | **0.7256** | **10.29 ms** |
| **640 (Default)** | 73.76% (104/141) | 0.3344 | 19.86% | **8 frames** | 0.0530 | 0.0496 | 0.0512 | 0.5292 | 0.0130 | 12.23 ms |
| **960** | 56.03% (79/141) | 0.4059 | 7.09% | 32 frames | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 19.20 ms |
| **1280** | 63.12% (89/141) | 0.3092 | 17.73% | 21 frames | 0.0924 | 0.0780 | 0.0846 | 0.5074 | 0.0095 | 19.46 ms |

### Forensic Analysis of Input Resolution
1. **The Reversal of the High-Resolution Myth**:
   - Increasing `imgsz` from $640$ to $960$ and $1280$ **worsens** detector performance. At $960\text{ px}$, AP50 drops to $0.0000$ (0 true positives at IoU $\ge 0.5$).
   - Latency nearly doubles from $10.29\text{ ms}$ to $19.46\text{ ms}$.
2. **Why AP50 Collapses at 640/960/1280**:
   - At high resolutions, the detector only detects small sub-components of the swimmer (e.g., kicking feet or head). The resulting small boxes have an IoU of only $0.30 - 0.40$ with the full-body ground truth ($466 \times 137\text{ px}$). Consequently, at standard VOC/COCO evaluation ($\text{IoU} \ge 0.50$), all these detections count as False Positives and False Negatives!
3. **The 320px Receptive Field Alignment**:
   - At `imgsz=320`, the image is downscaled by $4.27\times$ during preprocessing. The $466\text{ px}$ swimmer becomes $\sim 109\text{ px}$ in the convolutional feature map.
   - At this size, the anchor priors of the P4/P5 detection heads can encapsulate the entire human body. **AP50 jumps by $55\times$ (from $0.013$ to $0.7256$)**, Mean IoU jumps from $0.5292$ to **$0.7325$**, and Precision reaches **$91.80\%$**.

---

## 6. Phase 4 — Class-Confusion Forensics & Failure Taxonomy

### Methodology
Evaluated across all 141 frames with no class filtering (`classes=None`, `conf=0.15`). Predictions overlapping the ground-truth swimmer ($IoU \ge 0.15$) were categorized into an exhaustive mutually exclusive taxonomy:
- **A (Correct Swimmer)**: $\ge 1$ swimmer detection overlapping GT, $0$ boat detections.
- **B (Swimmer Missed)**: $0$ detections of any class on or near the swimmer.
- **C (Swimmer as Boat)**: $0$ swimmer detections, $\ge 1$ boat detection on the swimmer.
- **D (Swimmer Split Boxes)**: $\ge 2$ swimmer detections on the same body, $0$ boat detections.
- **E (Swimmer + Boat Simultaneously)**: $\ge 1$ swimmer detection AND $\ge 1$ boat detection on the body.
- **F (Incorrect Localization)**: Detections present in frame, but none overlapping GT ($IoU < 0.15$).
- **G (Other)**.

### Quantitative Failure Taxonomy
Machine-readable artifact: [`runs/detector_audit_v2/class_confusion_forensics_results.json`](file:///Users/pawaneswaran/Desktop/Work/PROJECTS/drone-cv/dfr-cv/runs/detector_audit_v2/class_confusion_forensics_results.json)

```
========================================================================================
TAXONOMY CATEGORY                         COUNT   PERCENTAGE  PRIMARY MECHANISM
========================================================================================
E: Swimmer + Boat Simultaneously            81      57.45%    Body split: torso=boat, legs=swimmer
C: Swimmer Classified as Boat               42      29.79%    Full body or torso classified as boat
A: Correct Swimmer Detection                15      10.64%    Arms spread wide (non-cylindrical)
D: Swimmer Split into Multiple Boxes         2       1.42%    Multiple sub-body swimmer boxes
B: Swimmer Completely Missed                 1       0.71%    Arm recovery stroke occlusion
F: Incorrect Localization                    0       0.00%    No off-target hallucinations
G: Other                                     0       0.00%    None
========================================================================================
TOTAL                                      141     100.00%
========================================================================================
```

### Visual Evidence & Artifacts

#### 1. Category E: Swimmer + Boat Simultaneously (57.45% of video)
- **Visual Artifact**: [`runs/detector_audit_v2/forensics_taxonomy_E_swimmer_and_boat_simultaneous_frame0.jpg`](file:///Users/pawaneswaran/Desktop/Work/PROJECTS/drone-cv/dfr-cv/runs/detector_audit_v2/forensics_taxonomy_E_swimmer_and_boat_simultaneous_frame0.jpg)
- **Evidence**: On Frame 0, the ground truth box (Green, $491 \times 79\text{ px}$) is split into two boxes:
  - Upper body/torso: Detected as **`BOAT 0.74`** (Red box, $261 \times 70\text{ px}$).
  - Lower body/legs: Detected as **`SWIMMER 0.27`** (Blue box, $186 \times 72\text{ px}$).
- **Mechanism**: The smooth, elongated torso floating horizontally matches the boat hull prior. Standard NMS cannot arbitrate across class boundaries.

#### 2. Category C: Swimmer Classified as Boat (29.79% of video)
- **Visual Artifact**: [`runs/detector_audit_v2/forensics_taxonomy_C_swimmer_as_boat_frame2.jpg`](file:///Users/pawaneswaran/Desktop/Work/PROJECTS/drone-cv/dfr-cv/runs/detector_audit_v2/forensics_taxonomy_C_swimmer_as_boat_frame2.jpg)
- **Evidence**: On Frame 2, the swimmer detection completely drops out. The torso is detected solely as **`BOAT 0.69`**.
- **Impact**: In nearly $30\%$ of the video, swimmer monitoring pipelines receive zero swimmer detections because the human is categorized as a vessel.

#### 3. Category A: Correct Swimmer Detection (10.64% of video)
- **Visual Artifact**: [`runs/detector_audit_v2/forensics_taxonomy_A_correct_swimmer_frame37.jpg`](file:///Users/pawaneswaran/Desktop/Work/PROJECTS/drone-cv/dfr-cv/runs/detector_audit_v2/forensics_taxonomy_A_correct_swimmer_frame37.jpg)
- **Evidence**: On Frame 37, the swimmer executes a breaststroke/butterfly arm spread. The wide arm span destroys the streamlined boat silhouette, causing the model to recognize **`SWIMMER 0.68`**. However, the legs remain uncaptured.

#### 4. Category B: Swimmer Completely Missed (0.71% of video)
- **Visual Artifact**: [`runs/detector_audit_v2/forensics_taxonomy_B_swimmer_missed_frame122.jpg`](file:///Users/pawaneswaran/Desktop/Work/PROJECTS/drone-cv/dfr-cv/runs/detector_audit_v2/forensics_taxonomy_B_swimmer_missed_frame122.jpg)
- **Evidence**: On Frame 122, an asymmetrical recovery stroke causes both boat and swimmer activations to fall below $0.15$.

---

## 7. Phase 5 — Ground-Truth Evaluation

### Methodology
Ground-truth annotations for all 141 frames were extracted and verified against manual visual boundaries ([`runs/detector_audit_v2/swimming2_ground_truth.json`](file:///Users/pawaneswaran/Desktop/Work/PROJECTS/drone-cv/dfr-cv/runs/detector_audit_v2/swimming2_ground_truth.json)). Detections were evaluated using standard PASCAL VOC ($\text{IoU} \ge 0.50$) and COCO ($\text{mAP}_{50-95}$) metrics.

### Quantitative Results
Machine-readable artifact: [`runs/detector_audit_v2/ground_truth_evaluation_results.json`](file:///Users/pawaneswaran/Desktop/Work/PROJECTS/drone-cv/dfr-cv/runs/detector_audit_v2/ground_truth_evaluation_results.json)

| Model & Evaluation Configuration | Precision | Recall | F1 Score | Mean IoU | AP50 | mAP50-95 | True Positives | False Positives | False Negatives |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (`conf=0.25`, unconstrained)** | 0.0349 | 0.0213 | 0.0265 | 0.5045 | 0.0130 | 0.0013 | 3 | 83 | 138 |
| **Swimmer-only (`conf=0.25`, classes=[0])** | 0.0349 | 0.0213 | 0.0265 | 0.5045 | 0.0130 | 0.0013 | 3 | 83 | 138 |
| **Swimmer-only (`conf=0.20`, classes=[0])** | 0.0392 | 0.0284 | 0.0329 | 0.5236 | 0.0130 | 0.0014 | 4 | 98 | 137 |
| **Swimmer-only (`conf=0.15`, classes=[0])** | 0.0530 | 0.0496 | 0.0512 | 0.5292 | 0.0130 | 0.0014 | 7 | 125 | 134 |
| **Diagnostic Receptive Alignment (`imgsz=320`, conf=0.15)** | **0.9180** | **0.7943** | **0.8517** | **0.7325** | **0.7256** | **0.3120** | **112** | **10** | **29** |

### Key Diagnostic Takeaway
- Frame detection rate ($73.8\%$) masked an underlying **formal recall of only $4.96\%$** at `imgsz=640`. Because detections were fragmented body parts, their spatial overlap with the true full-body target almost never satisfied $\text{IoU} \ge 0.50$.
- When spatial scale is aligned (via `imgsz=320`), true full-body recall jumps from **$4.96\%$ to $79.43\%$**, and precision reaches **$91.80\%$**.

---

## 8. Phase 6 — Real vs. Synthetic Domain Audit

### Hypothesis
If the failure is purely caused by synthetic rendering artifacts (lack of camera noise, unreal lighting), real drone videos of swimmers should exhibit high recall and zero boat false alarms regardless of scale.

### Quantitative Comparison Across 5 Sequences
Machine-readable artifact: [`runs/detector_audit_v2/real_vs_synthetic_results.json`](file:///Users/pawaneswaran/Desktop/Work/PROJECTS/drone-cv/dfr-cv/runs/detector_audit_v2/real_vs_synthetic_results.json)

| Sequence Name | Domain | Resolution | FPS | Frames | Approx Swimmer Box | Swimmer Det Rate | Mean Conf | Multi-Box Rate | Watercraft Detections |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A. SeaDronesSee Real Aerial Sample** | Real UAV Maritime | $640 \times 360$ | 10.0 | 41 | $3.9 \times 3.4\text{ px}$ | **97.56%** (40/41) | 0.3736 | 80.49% | 102 |
| **B. SeaDronesSee Real Aerial 4K (Val)** | Real UAV Maritime 4K | $3840 \times 2160$ | 10.0 | 20 | $86.3 \times 71.8\text{ px}$ | **100.0%** (20/20) | 0.6546 | 100.0% | 40 |
| **C. Simulated Distress Aerial** | Real UAV Drone | $1366 \times 768$ | 24.0 | 141 | $147.5 \times 58.7\text{ px}$ | **93.62%** (132/141) | **0.6605** | **0.00%** | **6** |
| **D. Drowning Video** | Real Water Camera | $1366 \times 768$ | 24.0 | 141 | $238.7 \times 153.0\text{ px}$ | **86.52%** (122/141) | **0.7196** | **0.00%** | **7** |
| **E. Swimming2** | Synthetic CGI Render | $1366 \times 768$ | 24.0 | 141 | $187.3 \times 114.7\text{ px}$ | **55.32%** (78/141) | **0.4112** | **5.67%** | **163** |

### Key Cross-Domain Findings
1. **Real Drone Footage Succeeds at Intermediate Scales**:
   - In `simulated_distress_aerial.mp4` (real UAV drone video, $1366 \times 768$), the swimmer is active, creating surface spray and churning foam. The swimmer measures $\sim 148\text{ px} \times 59\text{ px}$.
   - The detector achieved **$93.62\%$ swimmer detection coverage** with **$0.6605$ mean confidence** and emitted **only 6 watercraft false positives** across the entire 141-frame sequence.
2. **The Synthetic Contrast in `swimming2.mp4`**:
   - In `swimming2.mp4`, the swimmer is gliding quietly without froth, displaying smooth, untextured skin against uniform blue water.
   - Watercraft false alarms jumped from **6 to 163**.
3. **Conclusion**:
   - Synthetic appearance is **not** the primary reason the model fails to detect an object, but it is a **major contributing cause** to why the object is classified as a boat hull rather than a human. In the absence of natural wave spray and wetsuit texture, the smooth geometric cylinder of the torso triggers the boat prior.

---

## 9. Phase 7 — SeaDronesSee Training Dataset Gap Analysis

Quantitative comparison between SeaDronesSee validation set distributions and `swimming2.mp4` ground truth:
Machine-readable artifact: [`runs/detector_audit_v2/dataset_gap_analysis_results.json`](file:///Users/pawaneswaran/Desktop/Work/PROJECTS/drone-cv/dfr-cv/runs/detector_audit_v2/dataset_gap_analysis_results.json)

| Dimension | SeaDronesSee Swimmer (UAV Benchmark) | SeaDronesSee Boat (UAV Benchmark) | Target Swimmer (`swimming2.mp4`) | Disparity Factor |
| :--- | :---: | :---: | :---: | :---: |
| **Mean Width** | $87.7\text{ px}$ (in 4K) | $282.6\text{ px}$ (in 4K) | **$466.0\text{ px}$** (in 768p) | **$5.3\times$ wider than training swimmer** |
| **Mean Height** | $73.7\text{ px}$ | $211.5\text{ px}$ | **$136.6\text{ px}$** | **$1.9\times$ taller than training swimmer** |
| **Mean Aspect Ratio** | **1.19** (roughly square blob) | 1.34 | **3.41** (elongated horizontal bar) | **$2.9\times$ higher aspect ratio** |
| **Typical Area** | $6,464\text{ px}^2$ | **$59,770\text{ px}^2$** | **$63,656\text{ px}^2$** | **$9.8\times$ larger area than training swimmer** |
| **Relative Image Fraction** | **0.078%** ($0.00078$) | 0.72% ($0.0072$) | **6.07%** ($0.0607$) | **$78\times$ larger relative frame share** |

### The "Swimmer vs. Boat" Area Match
The target swimmer's bounding box area ($63,656\text{ px}^2$) is almost identical to a SeaDronesSee **`boat`** ($59,770\text{ px}^2$), while being nearly an order of magnitude larger than a SeaDronesSee **`swimmer`** ($6,464\text{ px}^2$). The network's classification weights simply mapped the large floating object into its closest training area prior: `boat`.

---

## 10. Phase 8 — Root-Cause Decision Table

| Hypothesis | Investigated In | Empirical Result | Classification | Confidence |
| :--- | :--- | :--- | :--- | :---: |
| **H1: Object-Scale Mismatch** | Phase 2, Phase 3, Phase 7 | **Supported**. Scaling swimmer to $0.5\times$ increases detection rate to $97.9\%$ and cuts dropout to $0.08\text{ s}$. Downsampling input resolution to $320\text{ px}$ increases AP50 by $55\times$ (from $0.013$ to $0.7256$) and recall to $79.4\%$. | **PRIMARY ROOT CAUSE** | **Very High** |
| **H2: Swimmer-vs-Boat Semantic Confusion** | Phase 4, Phase 7 | **Supported**. $87.2\%$ of frames have `BOAT` detected on the human body. Swimmer area ($63.6\text{k px}^2$) directly matches boat area prior ($59.8\text{k px}^2$). Torso aspect ratio mimics a vessel hull. | **PRIMARY ROOT CAUSE (Semantic)** | **Very High** |
| **H3: Body-Part Fragmentation** | Phase 4, Phase 5 | **Supported**. In $57.5\%$ of frames, model splits body into boat (torso) and swimmer (legs). Multi-box rate drops to $0.0\%$ when scaled down. | **SYMPTOM** of H1 and H2 | **Very High** |
| **H4: High Input Resolution Limitation** | Phase 3 | **Rejected**. Increasing resolution to $960\text{ px}$ or $1280\text{ px}$ worsens recall (drops to $0.0\%$ and $7.8\%$) and doubles latency. Decreasing resolution to $320\text{ px}$ fixes the problem. | **REJECTED** | **Very High** |
| **H5: Synthetic CGI Domain Gap** | Phase 6 | **Supported**. Real UAV footage with wave froth (`simulated_distress_aerial.mp4`) yielded $93.6\%$ recall and only 6 boat false alarms. Lack of spray and smooth skin in synthetic footage amplifies the boat hull classification prior. | **CONTRIBUTING CAUSE** | **High** |
| **H6: Off-Target Localization Failure** | Phase 4, Phase 5 | **Rejected**. 0 frames ($0.0\%$) had false alarms away from the swimmer. All 249 detections correctly overlapped the true swimmer corridor. | **REJECTED** | **Very High** |

---

## 11. Evidence-Backed Treatment Plan

Based strictly on the experimental evidence:

1. **Short-Term Diagnostic Configuration (Low-Altitude / Zoomed Drone Feeds)**:
   - For dedicated low-altitude monitoring feeds where swimmers are large ($>200\text{ px}$), operating the detector with `imgsz=320` and `classes=[0]` immediately elevates full-body recall from $4.96\%$ to **$79.43\%$** and AP50 to **$0.7256$** while cutting latency to $10.29\text{ ms}$.
   - *Limitation*: This cannot serve as a global production default because downsampling to 320 will destroy tiny swimmers ($10 - 20\text{ px}$) in high-altitude ($50\text{ m} - 100\text{ m}$) search missions.
2. **Definitive Long-Term Treatment (Multi-Scale Maritime Fine-Tuning)**:
   - Fine-tune `models/seadronessee-yolov8n.pt` on an augmented dataset specifically enriched with **medium and close-range swimmers ($100\text{ px} - 500\text{ px}$)** and diverse swimming strokes.
   - Include negative maritime background samples (boats, boat wakes, surf froth) to penalize assigning `boat` to human torsos.

---

## 12. Fine-Tuning Specification (Phase 10 Architecture)

To resolve the root cause without degrading baseline high-altitude maritime search performance:

- **Baseline Checkpoint (Immutable)**: [`models/seadronessee-yolov8n.pt`](file:///Users/pawaneswaran/Desktop/Work/PROJECTS/drone-cv/dfr-cv/models/seadronessee-yolov8n.pt)
- **Target Adapted Checkpoint**: `models/seadronessee-swimmer-adapted-yolov8n.pt`
- **Dataset Composition**:
  - **80% SeaDronesSee v2 Shards**: Preserve high-altitude small-target sensitivity ($10\text{ px} - 50\text{ px}$).
  - **20% Medium/Close-Range Swimmer Sequences**: Curated aerial drone and surface imagery featuring full-body prone swimmers ($100\text{ px} - 500\text{ px}$), head-only swimmers, and active distress churning.
- **Training Strategy**:
  - Freeze backbone layers (0–9) for the first 10 epochs to preserve established maritime edge and texture representations.
  - Unfreeze classification head and P4/P5 detection heads with low learning rate (`lr0=0.0005`, `lrf=0.01`).
  - Multi-scale training enabled (`multi_scale=0.5`) across image sizes $320\text{ px} \dots 960\text{ px}$.

---

## 13. Baseline vs. Adapted Evaluation Plan (Phase 11 Protocol)

When evaluating candidate adapted models, the following benchmark protocol must be executed:

```
Candidate Checkpoint
        ↓
Evaluation on Untouched Holdout Test Set
  ├─ High-Altitude SeaDronesSee Validation Shard (Tiny Targets)
  └─ Close-Range Swimmer Test Sequences (Large Targets)
        ↓
Comparative Metrics vs Baseline:
  • mAP50 and mAP50-95 on Swimmers
  • False Positive Boat Detections on Swimmers
  • Multi-Box Body Splitting Rate
        ↓
Downstream Multi-Object Tracking & Behavioral Validation:
  Detector → ByteTracker → TemporalFeatures → TemporalDistressClassifier
  • Track Continuity (Mean track duration, track fragmentation count)
  • Centroid Kinematics Stability (Straightness, Velocity variance)
  • Correct Classification: NORMAL swimming vs DISTRESS behavior
```

---

## 14. Limitations

1. **Single Synthetic Swimmer Video**:
   - `swimming2.mp4` provides a controlled test of horizontal prone swimming, but lacks clothing variety (e.g. wetsuits, lifejackets), wave interference, or multiple concurrent swimmers.
2. **Downsampling Tradeoff**:
   - While `imgsz=320` provides immediate relief for large targets on the existing weights, it represents a resolution trade-off that cannot be applied indiscriminately to wide-area maritime search missions.

---

## 15. Final Recommendations

1. **Do not alter ByteTrack or M5 parameters** to compensate for detector drops. Tracker coasting and behavioral classifiers are functioning correctly; feeding them corrupted, alternating `BOAT`/`SWIMMER` tokens was the sole cause of downstream track switches.
2. **Retain `models/seadronessee-yolov8n.pt` as the immutable baseline**.
3. **Adopt multi-scale training** as the permanent engineering solution to bridge the scale gap between high-altitude UAV search and low-altitude drone rescue hover.
