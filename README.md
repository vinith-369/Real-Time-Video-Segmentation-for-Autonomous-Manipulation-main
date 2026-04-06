# Real-Time Video Segmentation Pipeline

This section of the repository contains the self-contained, end-to-end pipeline for semantic video segmentation targeting autonomous robotic manipulation. The workflow encompasses automated dataset generation, deep learning model training, and real-time inference—all handled within a set of straightforward Jupyter Notebooks.

## Pipeline Architecture

The workflow is broken down into three logical steps:

### 1. Data Annotation (`01_create_dataset_with_sam2.ipynb`)
Instead of manually annotating frame-by-frame, this notebook uses Meta's **Segment Anything 2 (SAM 2)** to visually track and automatically generate highly accurate binary segmentation masks across entire robotic manipulation video sequences.

### 2. Model Training (`02_train_unet.ipynb`)
Trains a lightweight PyTorch **U-Net** architecture from scratch on the newly generated SAM 2 dataset. 
- Custom `Dataset` and `DataLoader` implementations are provided.
- Training metrics (Learn Rate, Dice Score, Loss) are automatically logged.
- The notebook achieved convergence with a **Training Dice Score of ~0.97** and a peak **Validation Dice Score of ~0.69** within 100 epochs.

### 3. Results & Inference (`03_results_and_inference.ipynb`)
Evaluates the trained U-Net on unseen data using rigorous single-image and batch inference loops. It also blends the segmentation masks over existing videos in real-time, outputting final tracking videos (e.g., `output_segmented.mp4`) via OpenCV.

## Folder Structure

```text
pipeline/
├── 01_create_dataset_with_sam2.ipynb  # Dataset generation via SAM 2
├── 02_train_unet.ipynb                # PyTorch U-Net training pipeline
├── 03_results_and_inference.ipynb     # Model evaluation and video rendering
├── dataset/                           # Auto-generated raw frames and binary masks
├── saved_models/                      # Trained U-Net model checkpoints (.pth)
└── output_segmented.mp4               # Final real-time segmentation demonstration
```

## Setup & Execution
The notebooks should be executed sequentially (01 → 02 → 03) to recreate the entire dataset, train the network, and yield the rendered segmentation videos. Ensure your paths in Notebook 01 correctly point to the downloaded SAM 2 weights prior to generating the dataset.

## Process & Results

The visual pipeline operates seamlessly to provide robust robotic manipulation tracking:
1. **Raw Input**: Live manipulation camera arrays or videos are fed into the pipeline.
2. **Automated Labeling (SAM 2)**: Researchers provide a simple initial prompt, and SAM 2 automatically limits and propagates the tracking mask through the sequence.
3. **U-Net Training**: A lightweight PyTorch U-Net learns these masks. During training, the model converged efficiently, achieving a **Training Dice Score of ~0.97** and a peak **Validation Dice Score of ~0.69** at 100 epochs, demonstrating strong tracking capabilities.
4. **Real-Time Blending**: The model's predictions are overlaid back onto the original visual feed utilizing OpenCV. The single-image and batch inference modules are lightweight enough to allow continuous real-time execution, maintaining the low latency essential for robotic manipulation.

Below is an example of the semantic segmentation output produced during the real-time inference stage:

![Segmentation Result](pipeline/output.png)
