# Real-Time Video Segmentation for Autonomous Robotic Manipulation

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python" />
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch" />
  <img src="https://img.shields.io/badge/SAM2-Meta%20AI-0467DF?style=for-the-badge" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" />
</p>

A **2-stage deep learning pipeline** that enables autonomous robotic manipulators to perform real-time object segmentation in video streams. It uses **Meta's SAM2** (Segment Anything Model 2) to automatically generate high-quality training labels, then distills that knowledge into a lightweight **U-Net** that runs at real-time inference speed.

---

## 🔍 Overview

```
Raw Video
    │
    ▼
[Notebook 01] ──► SAM2 Auto-Labeling ──► Dataset (images + binary masks)
    │
    ▼
[Notebook 02] ──► U-Net Training ──► Trained Checkpoint (.pth)
    │
    ▼
[Notebook 03] ──► Real-Time Inference ──► Segmented Video Output + FPS Metrics
```

The key insight: SAM2 is highly accurate but slow. We use it **once** to label data, then train a fast U-Net to replicate its output — achieving near-SAM2 quality at real-time speed suitable for robotic control loops.

---

## ✨ Features

- 🎯 **Interactive annotation** — click on an object in frame 1, SAM2 propagates masks to all remaining frames automatically
- 🚀 **Real-time inference** — lightweight U-Net runs fast enough for live robot control
- 📊 **Full evaluation** — Dice Score, Pixel Accuracy, FPS benchmarking with plots
- 🎥 **Video output** — saves segmented video with mask overlay
- 🧩 **Modular pipeline** — 3 self-contained Jupyter notebooks, easy to adapt

---

## 📁 Repository Structure

```
Real-Time-Video-Segmentation-for-Autonomous-Manipulation-main/
│
├── 01_create_dataset_with_sam2.ipynb     # Stage 1: Auto-labeling with SAM2
├── 02_train_unet.ipynb                   # Stage 2: U-Net training & evaluation
├── 03_results_and_inference.ipynb        # Stage 3: Real-time inference & benchmarking
│
├── saved_models/
│   ├── unet_trained.pth                  # Primary trained model checkpoint
│   └── unet_trained1.pth                 # Alternate checkpoint
│
├── requirements.txt                      # Python dependencies
├── Literature_Survey_RealTime_Video_Segmentation.docx.pdf
└── Real-Time-Video-Segmentation-for-Autonomous-Robotic-Manipulation.pptx
```

---

## ⚙️ Setup & Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended) or CPU
- [SAM2 repository](https://github.com/facebookresearch/segment-anything-2) cloned locally

### 1. Clone this repository

```bash
git clone https://github.com/adhacks541/Real-Time-Video-Segmentation-for-Autonomous-Manipulation-main.git
cd Real-Time-Video-Segmentation-for-Autonomous-Manipulation-main
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up SAM2

```bash
# Clone SAM2 into a local directory
git clone https://github.com/facebookresearch/segment-anything-2.git sam2_repo
cd sam2_repo
pip install -e .

# Download a SAM2 checkpoint (e.g., sam2_hiera_large.pt)
# https://github.com/facebookresearch/segment-anything-2#model-checkpoints
```

---

## 🚀 Usage

### Stage 1 — Create Dataset with SAM2

Open `01_create_dataset_with_sam2.ipynb` and update the **CONFIGURATION** section:

```python
VIDEO_PATH     = "path/to/your/input_video.mp4"
SAM2_REPO_PATH = "path/to/sam2_repo"
CHECKPOINT     = "path/to/sam2_hiera_large.pt"
OUTPUT_DIR     = "dataset/"
```

Run all cells. When prompted, **click on the object** you want to segment in the first frame. SAM2 will propagate the mask to all frames automatically.

---

### Stage 2 — Train the U-Net

Open `02_train_unet.ipynb` and update the config:

```python
IMAGE_DIR       = "dataset/raw_images"
MASK_DIR        = "dataset/masks"
CHECKPOINT_PATH = "saved_models/unet_trained.pth"
NUM_EPOCHS      = 50
BATCH_SIZE      = 8
LEARNING_RATE   = 1e-4
INPUT_SIZE      = (256, 256)
```

Run all cells to train, evaluate, and save the model.

---

### Stage 3 — Real-Time Inference

Open `03_results_and_inference.ipynb` and set:

```python
CHECKPOINT_PATH  = "saved_models/unet_trained.pth"
TEST_IMAGE_DIR   = "dataset/raw_images"
VIDEO_PATH       = "path/to/test_video.mp4"
OUTPUT_VIDEO_PATH = "output_segmented.mp4"
```

Run all cells to see:
- Single & batch image predictions
- Full video segmentation with mask overlay
- FPS performance plots

---

## 🏗️ Model Architecture

The U-Net uses a lightweight **encoder-decoder** design with skip connections:

```
Input (3, H, W)
    │
    ▼
[Encoder Block 1] → 64 channels  ──────────────────────────────┐
    │ MaxPool                                                   │ Skip
    ▼                                                           │
[Encoder Block 2] → 128 channels ─────────────────────────┐   │
    │ MaxPool                                               │   │
    ▼                                                       │   │
[Bottleneck]       → 256 channels                          │   │
    │ Upsample                                              │   │
    ▼                                                       │   │
[Decoder Block 1] ← concat ←──────────────────────────────┘   │
    │ Upsample                                                  │
    ▼                                                           │
[Decoder Block 2] ← concat ←──────────────────────────────────┘
    │
    ▼
Output (1, H, W) → Sigmoid → Binary Mask
```

- **Loss:** Binary Cross-Entropy
- **Optimizer:** Adam
- **Metrics:** Dice Score, Pixel Accuracy

---

## 📈 Results

| Metric | Value |
|--------|-------|
| Dice Score (val) | _run notebook to get your results_ |
| Pixel Accuracy (val) | _run notebook to get your results_ |
| Inference Speed | _device-dependent — see Notebook 03_ |

---

## 🔮 Future Work

- [ ] Add data augmentation (flips, color jitter, rotation)
- [ ] Try Dice Loss / Focal Loss for better mask boundary accuracy
- [ ] Deeper U-Net (3–4 encoder blocks) for complex scenes
- [ ] ONNX export for edge deployment (NVIDIA Jetson, etc.)
- [ ] ROS2 integration node to publish masks as topics
- [ ] Multi-class / multi-object segmentation support
- [ ] Live webcam inference demo

---

## 📚 References

- [Segment Anything Model 2 (SAM2) — Meta AI](https://github.com/facebookresearch/segment-anything-2)
- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
- Literature Survey: see `Literature_Survey_RealTime_Video_Segmentation.docx.pdf`

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m 'feat: add your feature'`
4. Push: `git push origin feature/your-feature`
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License.

---

<p align="center">Made with ❤️ for autonomous robotics research</p>
