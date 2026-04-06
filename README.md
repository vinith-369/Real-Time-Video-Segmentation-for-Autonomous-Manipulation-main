# Real-Time Video Segmentation for Autonomous Manipulation

This repository contains tools, models, and pipelines for achieving real-time semantic video segmentation, designed specifically to aid autonomous robotic manipulation. By leveraging a variety of techniques—from classical computer vision baselines to modern deep learning architectures and foundation models—the project builds a comprehensive framework for scene understanding and object tracking in manipulation tasks.

## Table of Contents
- [Overview](#overview)
- [Project Architecture](#project-architecture)
- [Key Components & Features](#key-components--features)
  - [1. Data Annotation with SAM 2](#1-data-annotation-with-sam-2)
  - [2. Deep Learning Segmentation (U-Net)](#2-deep-learning-segmentation-u-net)
  - [3. Classical CV Baseline](#3-classical-cv-baseline)
  - [4. Real-Time Inference](#4-real-time-inference)
- [Repository Structure](#repository-structure)
- [Dependencies & Setup](#dependencies--setup)

## Overview

In autonomous manipulation setups, robots visually observe a scene, track moving entities (like end-effectors, tools, and objects), and act based on these perceptions. This repository tackles the perception side by focusing on extracting high-quality segmentation masks from streaming video data. 

## Project Architecture

The core pipeline operates in three main stages:
1. **Automated Dataset Generation:** Generating ground-truth masks by prompting [Segment Anything 2 (SAM 2)](https://github.com/facebookresearch/segment-anything-2) over short segments of video.
2. **Model Training:** Training a custom PyTorch `U-Net` on the newly generated dataset to run faster and with less overhead compared to a full foundation model.
3. **Inference & Benchmarking:** Feeding new live camera inputs/video files into the optimized pipelines to measure real-time throughput and accuracy.

## Key Components & Features

### 1. Data Annotation with SAM 2
*Found in `pipeline/` and `sam2_repo/`*  
Instead of spending hours manually annotating video frames, we utilize Meta's SAM 2 to automatically propagate segmentation masks through video. By selecting a Region of Interest (ROI) in an initial frame, SAM 2's camera predictor tracks and segments the desired objects across the entire sequence.

### 2. Deep Learning Segmentation (U-Net)
*Found in `UNet/` and `pipeline/`*  
A lighter architecture for segmentation that prioritizes high frames-per-second (fps) performance. 
- Fully-featured PyTorch implementation.
- Uses `Weights & Biases (wandb)` for logging metrics like Dice scores, losses, and learning rates.
- Includes a dedicated inference script to preview single image predictions.

### 3. Classical CV Baseline
*Found in `Classical CV Baseline/`*  
To establish a baseline for segmentation metrics and inference speed, a deterministic non-learning approach is provided via `object_mask.py`. This acts as a fallback or a performance benchmark for the neural networks. 

### 4. Real-Time Inference
*Found in `scripts/`*  
Scripts such as `rt_seg.py` provide a streamlined application of video prediction without the heavy overhead of training setups. Leveraging `OpenCV` (cv2), it captures video input, parses through the segmentation models, and outputs a blended video with overlaid tracking masks.

## Results

- **Segmentation Quality**: Our custom PyTorch U-Net successfully learns tracking masks from the SAM2 datasets. Over 100 training epochs, the model achieved a **Training Dice Score of ~0.97** (Train Loss: 0.012). Validation Dice scores reached a peak of **~0.69** during early convergence.
- **Real-Time Performance**: The single-image and batch inference modules are lightweight enough to allow continuous real-time video blending via OpenCV, proving its suitability for dynamic robotic manipulation tasks where low latency is critical.

## Repository Structure

```text
.
├── Classical CV Baseline/  # Traditional CV techniques for baseline comparisons
├── Data/                   # Datasets, ground truth masks, and raw images
├── Media/                  # Input manipulation videos and output recordings
├── UNet/                   # PyTorch codes for custom U-Net segmentation
│   ├── main.py             # Main trainer pipeline for U-Net
│   ├── utils.py            # Model architecture, dataloaders, and transforms
│   ├── data_processing.ipynb 
│   └── Saved Models/       # Checkpoints and weights
├── pipeline/               # Step-by-step Jupyter Notebooks for the whole system
│   ├── 01_create_dataset_with_sam2.ipynb
│   ├── 02_train_unet.ipynb
│   └── 03_results_and_inference.ipynb
├── sam2_repo/              # Segment Anything Model 2 codebase & model weights
└── scripts/                # Real-time tracking and demonstration scripts
```

## Dependencies & Setup

Major dependencies include:
- `torch`, `torchvision`, `torchaudio`
- `opencv-python`
- `numpy`, `matplotlib`, `Pillow`
- `wandb`
- `sam2` (from the local repository)

Ensure to configure your `sam2_repo/checkpoints` properly when invoking the automated masking scripts. Training configurations such as epoch count, logging, and dataset paths can be modified in `UNet/main.py`.
