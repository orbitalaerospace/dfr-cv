# M3 Drowning Model Audit

## Candidate Model

- **Repository**: `Z5cc/drowning-detection` (David Nicklaser, 2024)
- **Reference**: *"Drowning Detection via Real-Time Spatio-Temporal Action Localization"*
- **Core Candidates Investigated**:
  1. **`2Dseq` (Primary Candidate)**: 2D YOLO-Free backbone + multi-level ConvLSTM recurrent temporal reasoning + decoupled detection head.
  2. **`YOWOv2` (Baseline Reference)**: Dual-stream spatio-temporal action localization model (2D YOLO backbone for current frame + 3D CNN backbone over $T=32$ temporal frames + channel fusion encoder).
  3. **`2D3Dseq` & `3Dseq` (Ablations)**: 3D CNN backbones coupled with ConvLSTM layers (found inferior to `2Dseq` in speed and accuracy by author).

---

## Architecture

The investigated repository evaluates four model configurations, with **`2Dseq`** established as the author's primary and best-performing architecture:

```
[Frame t: 3 x 224 x 224]
        │
        ▼
┌──────────────────────────────────────────────┐
│ 2D Spatial Backbone (FreeYOLO / PAFPN-ELAN)  │
└──────────────────────────────────────────────┘
        │ P3 (stride 8)   │ P4 (stride 16)   │ P5 (stride 32)
        ▼                 ▼                  ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│ ConvLSTM L1  │   │ ConvLSTM L2  │   │ ConvLSTM L3  │  ◄── [Hidden States: h_{t-1}, c_{t-1}]
└──────────────┘   └──────────────┘   └──────────────┘
        │                 │                  │
        ▼                 ▼                  ▼
┌──────────────────────────────────────────────┐
│ Decoupled Detection Heads (Conf / Cls / Reg) │
└──────────────────────────────────────────────┘
        │
        ▼
Output Detections: Bboxes [4], Confidence [1], Class Logits [2] (swim vs. drown)
```

### Detailed Component Specifications

1. **2D Spatial Backbone**:
   - `yolo_free_nano`, `yolo_free_tiny`, or `yolo_free_large`.
   - Feature pyramid with 3 output scales: Strides 8 ($28 \times 28$), 16 ($14 \times 14$), and 32 ($7 \times 7$) at input resolution $224 \times 224$.
   - Pretrained on MS COCO.

2. **Recurrent Temporal Module (`ConvLSTM`)**:
   - 3 separate `ConvLSTM` modules attached at each pyramid scale level.
   - Tiny variant: 64 input channels, 64 hidden channels, $3 \times 3$ convolutional kernels, 2–3 stacked LSTM layers.
   - Medium variant: 256 input channels, 128 hidden channels, $3 \times 3$ convolutional kernels.
   - Retains hidden state $\mathbf{h}_t$ and cell state $\mathbf{c}_t$ across frames to model long-term temporal movement patterns.

3. **Decoupled Detection Heads**:
   - Non-shared heads per pyramid level for confidence (`conf_preds`), classification (`cls_preds`), and bounding box regression (`reg_preds`).
   - SimOTA dynamic label assignment during training.

---

## Input Requirements

- **Input Dimensions**: $224 \times 224$ pixels, 3 channels (RGB).
- **Data Preprocessing**:
  - Image resizing to $224 \times 224$.
  - Channel normalization to $[0.0, 1.0]$.
  - Standard color jitter, hue, saturation, and exposure augmentations applied only during training.
- **Batch Size**:
  - Training: Batch size 4–16.
  - Inference: Fixed to `batch_size = 1` because recurrent hidden states $\mathbf{h}_t, \mathbf{c}_t$ must be updated sequentially per continuous video stream.

---

## Temporal Window

- **`2Dseq` Temporal Handling**:
  - **Training Window**: Sequence length $S = 48$ frames sampled at temporal stride $S_{\text{distance}} = 3$ frames. Total temporal span = $48 \times 3 = 144$ video frames ($\approx 4.8$ seconds at 30 FPS).
  - **Inference Mode**: Sequential step-by-step processing ($S = 1$). Each incoming frame updates the ConvLSTM hidden state, allowing theoretically infinite temporal context until the hidden state is reset.
- **`YOWOv2` (Reference) Temporal Handling**:
  - Sliding temporal buffer of $T = 32$ frames sampled at temporal stride $T_{\text{distance}} = 6$ frames. Total temporal span = $32 \times 6 = 192$ video frames ($\approx 6.4$ seconds at 30 FPS).
- **Expected Input Frame Rate**: 30 FPS.

---

## Native Classes

The model ontology in the repository (`config/default_config.py` and `dataset_utils/LifeguardUtils.py`) is strictly binary:

| Native Class ID | Native Label Name | Semantic Definition in Repository |
|:---:|:---:|:---|
| `0` | `swim` | Individual swimming normally, playing, or afloat without distress. |
| `1` | `drown` | Individual exhibiting active drowning behavior (IDR), struggling, or submerged in distress. |

### Semantic Mapping vs. dfr-cv Domain Contract

| `Z5cc` Native Class | `dfr-cv` TargetClass | Mapping Status / Semantic Gap |
|:---:|:---:|:---|
| `swim` | `TargetClass.SWIMMER` (or `PERSON`) | **Approximate match**. Indicates an active person in water not in distress. |
| `drown` | `TargetClass.DISTRESS` / `TargetClass.DROWNING` | **Severe semantic leap if untreated**. In `dfr-cv`, `DISTRESS` is an early tracklet hypothesis and `DROWNING` is a confirmed multi-stage incident state. Direct single-frame classification from raw video conflates observation with operational incident confirmation. |

---

## Output Format

For each frame, the model outputs:
1. **Bounding Boxes**: Normalized tensor $[x_{\min}, y_{\min}, x_{\max}, y_{\max}] \in [0.0, 1.0]^4$.
2. **Confidence Scores**: Combined objectness and class confidence:
   $$\text{Score} = \sqrt{\sigma(\text{conf}) \cdot \sigma(\text{cls})} \in [0.0, 1.0]$$
3. **Class Predictions**: Label index $\in \{0, 1\}$ (`0: swim`, `1: drown`).
4. **State Tuple (in `2Dseq`)**: Updated hidden states `new_hidden_state` carrying the spatio-temporal recurrent representations for the next inference step.

---

## Dataset

- **Name**: `dataset_lifeguard`
- **Origin**: 82 public YouTube videos, mostly scraped from the YouTube channel *"Lifeguard Rescue"* (surveillance recordings of public swimming pools and wave pools).
- **Volume**: 111 video snippets cut from the 82 source videos.
- **Resolution**: $224 \times 224$ pixels per snippet.
- **Duration**: $\approx 25$ seconds per snippet ($\sim 750$ frames each), yielding $\approx 46$ minutes of total video.
- **Subjects**: Average of $\sim 3$ individuals per snippet.
- **Repository State**: **The actual video files and extracted frames are NOT distributed in the repository.** The repo contains only a download script (`LifeguardUtils.py`) that relies on `yt-dlp` to pull videos from YouTube.

---

## Training Labels

- **Format**: 91 Pascal VOC XML annotation files (`dataset_lifeguard/labels/*.xml`) exported from CVAT.
- **Label Granularity**: Frame-level bounding box annotations with class attributes `<box label="swim">` and `<box label="drown">`.
- **Annotation Method**: Bounding boxes manually annotated on every 6th frame; intermediate 5 frames generated via CVAT linear interpolation.
- **Label Quality**: Coarse bounding boxes around swimmers and drowning victims. Because interpolation is linear across 6 frames, bounding box tracking jitter occurs when swimmers change direction rapidly.

---

## Evaluation Protocol

- **Metric**: Frame-level Mean Average Precision (**Frame mAP**) at $\text{IoU} = 0.5$ using Pascal VOC 11-point interpolation.
- **Validation Strategy**: 9-fold cross-validation (`1_test.csv` through `9_test.csv`), each test fold containing 10 snippets ($\approx 90\%$ train, $10\%$ test).
- **Critical Evaluation Flaw (Data Leakage)**:
  - The split is generated by shuffling snippets (`df.sample(frac=1)`), **not by unique YouTube video ID**.
  - Multiple snippets originate from the exact same YouTube video (e.g., `5rxH_ELrwTU`, `I-EKqLdbysk`, `IDpdBu-je6E`).
  - As a result, snippets from the same video—sharing identical pool tiles, water clarity, lighting, camera geometry, and even the same swimmers—exist simultaneously in both training and test splits.
  - Reported validation results are consequently overly optimistic and vulnerable to scene memorization.

---

## Pretrained Weights

- **Availability**: **NONE AVAILABLE**.
- **Audit Findings**:
  - No `.pth` or `.pt` weight files are included in the repository or download archives.
  - The repository contains download URLs only for generic upstream backbones:
    - FreeYOLO COCO weights (`yolo_free_nano_coco.pth`, `yolo_free_large_coco.pth`).
    - 3D CNN Kinetics-400 weights (`kinetics_shufflenetv2_1.0x_RGB_16_best.pth`, `resnext-101-kinetics.pth`).
  - The final trained drowning model checkpoints referenced in the code (e.g., `yowo_v2_nano_epoch_7.pth`, `2Dseq_medium`) are absent.
- **Impact**: The model cannot be executed or evaluated out of the box without fully rebuilding the dataset and retraining from scratch.

---

## License / Provenance

| Component | License / Provenance | Status | Risk Level |
|:---|:---|:---:|:---:|
| **Repository Source Code** | No `LICENSE` file present. Upstream components (YOWOv2, ConvLSTM) are MIT, but author's code has no explicit license. | **UNCERTAIN** | Medium |
| **Pretrained Weights** | Not published / Missing. | **N/A** | None (weights do not exist) |
| **Dataset Source Material** | Scraped YouTube videos from third-party channel *"Lifeguard Rescue"* via `yt-dlp`. | **PROPRIETARY / NO LICENSE** | **CRITICAL** (Copyright infringement & YouTube TOS violation if used commercially) |

> [!WARNING]
> The dataset cannot be legally used in a commercial drone perception product. Training commercial production models on scraped YouTube lifeguard videos without licensing agreements creates severe copyright and intellectual property liability.

---

## Integration Difficulty

**Rating: HIGH**

1. **Architecture Mismatch**:
   - `2Dseq` is designed as a monolithic end-to-end detector operating on small, pre-cropped $224 \times 224$ images.
   - Feeding full-resolution aerial imagery ($1920 \times 1080$ or $4\text{K}$) through multi-scale ConvLSTM layers would exhaust edge GPU memory and drop inference speeds far below real-time.
2. **State Management Complexity**:
   - `2Dseq` maintains hidden states across the entire frame. If multiple swimmers are in a scene, the ConvLSTM models whole-scene background dynamics rather than individual target trajectories.
   - For an aerial drone system, temporal reasoning belongs at the **tracklet level** (per tracked person bounding box over time), not at the full-frame spatial backbone level.
3. **Absence of Weights**:
   - Because no trained weights exist, an "integration spike" cannot evaluate inference on real drone video without first training the network from scratch.

---

## Strengths

1. **Temporal Distinction**: Accurately recognizes that drowning cannot be determined from a single static image; temporal motion modeling is mathematically essential.
2. **Behavioral Problem Formulation**: Specifically tackles genuine behavioral ambiguities: separating the Instinctive Drowning Response from normal swimming strokes, climbing/descending in water, and breaststroke/crawl.
3. **`2Dseq` Computational Efficiency**: Demonstrates that recurrent 2D feature maps (`2Dseq`) run $3\times$ faster than heavy 3D spatio-temporal convolutions while achieving comparable or superior temporal discrimination.

---

## Limitations

1. **Missing Weights**: Complete blocker for immediate inference verification.
2. **Tiny Dataset**: 111 snippets ($\approx 46$ minutes) is several orders of magnitude smaller than required for robust computer vision in uncontrolled outdoor environments.
3. **Data Leakage in Evaluation**: Cross-validation split at snippet level rather than video level skews reported accuracy.
4. **No Tracklet Decoupling**: Conflates spatial object detection with temporal action recognition in a single monolithic network.

---

## Domain Gap

There is an enormous operational domain gap between the research setup and an autonomous aerial drone deployment:

| Dimension | `Z5cc` Research Dataset | Aerial Drone SAR Environment (`dfr-cv`) | Edge Operational Reality |
|:---|:---|:---|:---|
| **Platform & Perspective** | Fixed CCTV / hand-held poolside cameras; oblique to horizontal viewing angle ($\approx 0^\circ - 30^\circ$). | UAV-mounted gimbal camera; high-altitude oblique to pure nadir ($\approx 45^\circ - 90^\circ$). | Camera undergoes roll/pitch/yaw motions, gusts, platform vibration, and forward flight motion. |
| **Target Scale** | Large, prominent swimmer profiles occupying $15\% - 40\%$ of the $224 \times 224$ crop. | Tiny targets; person heads/shoulders often occupy only $10 \times 10$ to $30 \times 30$ pixels on a $1080\text{p}$ sensor ($<0.1\%$ of frame). | Feature representations from $224\times 224$ backbones lose tiny swimmer features entirely. |
| **Water Environment** | Chlorinated turquoise swimming pools and enclosed wave pools with distinct pool tiles and lane lines. | Open ocean, coastal surf, estuaries, rivers, lakes. | Dynamic waves, whitecaps, swell, sun glint, seafoam, murky water, and boat wakes. |
| **Distress Mechanics** | Pool drowning: vertical body position, head tilted back, lateral arm splashing in wave pool chop. | Open-water drowning: rip-current exhaustion, offshore drift, hypothermia passivity, submergence under wave sets. | Swimmer distress signatures in open ocean differ substantially from pool wave machine struggles. |
| **Compute Budget** | Desktop GPU (NVIDIA RTX series). | NVIDIA Jetson Orin Nano / NX ($10\text{W} - 25\text{W}$ power budget). | Monolithic 3D CNNs or dense ConvLSTMs cannot sustain 20+ FPS at high resolution. |

---

## Recommendation

### **Decision: 4. BLOCKED BY MISSING WEIGHTS/LICENSE/DATA**

### Justification:

1. **No Pretrained Weights**: The repository provides zero trained checkpoint weights for drowning vs. swimming behavior. We cannot run an inference spike on our test footage without training a model from scratch.
2. **Copyright & TOS Liability**: The training data consists of unlicenced, scraped YouTube videos from a third-party channel. It cannot legally serve as training or fine-tuning data for a commercial drone product.
3. **Data Leakage & Pool Bias**: The dataset is too small (111 snippets, $\sim 46$ min), pool-specific, and methodologically flawed by snippet-level train/test leakage.
4. **Architectural Incompatibility for Direct Integration**: Monolithic end-to-end full-frame ConvLSTMs do not fit high-resolution aerial search and rescue. 

### Correct Architectural Direction for `dfr-cv`:

Rather than forcing an end-to-end $224 \times 224$ monolithic model into our aerial pipeline:
1. **Retain Spatial Detector**: Use high-resolution YOLO (trained on aerial maritime data such as SeaDronesSee) to detect people/watercraft.
2. **Multi-Object Tracking (ByteTrack)**: Associate detections across frames to establish persistent person tracklets.
3. **Decoupled Tracklet Temporal Classifier**: Extract normalized bounding box motion kinematics (speed, displacement, aspect ratio variance, submergence frequency) and/or cropped patch sequence features per tracklet to classify **Normal Swimmer vs. Distress / Drowning Candidate**.

---

## Most Important Question

> **"Can this existing model genuinely distinguish swimming from drowning from temporal video information, and what evidence supports that conclusion?"**

### Direct Answer:
**Conceptually yes, but empirically unverified in our environment, legally blocked, and fundamentally unvalidated for aerial drone search and rescue.**

### Evidence Analysis:
1. **Temporal Mechanics are Sound**:
   - Swimming exhibits continuous directional displacement, horizontal aspect ratios, and rhythmic periodic arm/leg cycles.
   - Active drowning (the Instinctive Drowning Response) exhibits near-zero horizontal velocity, vertical orientation, rapid vertical displacement/bobbing, and sporadic arm flailing without propulsion.
   - The paper/repo proves that single-frame 2D spatial features alone fail to disambiguate ascending/descending or crawling from drowning, whereas adding temporal sequence modeling (ConvLSTM over $S=48$ frames) resolved these false positives in their pool test cases.
2. **Empirical Evidence in Repo is Compromised**:
   - The visual demos (GIFs) confirm that on poolside CCTV recordings of wave pools, `2Dseq` suppresses false drowning alarms caused by diving or regular swimming.
   - However, the quantitative evidence (Frame mAP across 9 folds) is compromised by train/test data leakage across snippets from identical YouTube source videos.
3. **Zero Aerial Evidence**:
   - There is zero evidence that this model or its learned features function on top-down aerial drone footage over open ocean or coastal waters.
4. **Execution Failure**:
   - Because no trained weights were published by the author, the model's performance cannot be independently verified or tested on real aerial video today.
