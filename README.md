# ForensicFlow
**A Tri-Modal Adaptive Network for Robust Deepfake Detection**  

**arXiv 2025** • **AUC 0.9752** on Celeb-DF (v2) • **15 epochs only**  
**Full-frame processing • No face cropping • Frequency branch • Attention-based temporal fusion**

[![Paper](https://img.shields.io/badge/arXiv-2511.XXXXX-b31b1b.svg)](https://arxiv.org/abs/2511.14554))
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org)

**Mohammad Romani**  
m.romani@modares.ac.ir • November 18, 2025  

---

## Why ForensicFlow stands out

- **Full-frame forensics** — deliberately avoids face cropping to preserve blending artifacts at facial boundaries (unlike 95% of existing methods)
- Explicit **frequency branch** that directly models periodic noise patterns typical of GAN-generated content
- **Attention-powered temporal pooling** — dynamically focuses on the most discriminative frames
- Lightweight yet SOTA: ConvNeXt-tiny + Swin Transformer-tiny backbones
- **No MTCNN / facenet-pytorch dependency** — pure, clean, full-frame design

> "Sometimes the biggest improvement comes from knowing what *not* to do."

## Results (Celeb-DF v2 validation)

| Method        | AUC     | F1-Score | Accuracy | Epochs |
|---------------|---------|----------|----------|--------|
| ForensicFlow  | **0.9752** | **0.9408** | **0.9208** | 15     |

95% CI (n_boot=1000): AUC [0.9636, 0.9848]

## Architecture

<img src="architecture.png" alt="ForensicFlow Architecture" width="100%"/>

## Quick Start

```bash
git clone https://github.com/mohammad-romani/ForensicFlow.git
cd ForensicFlow
pip install -r requirements.txt

# 1. Preprocess videos to .npz (run once)
python utils.py --data_path ./Celeb-DF-v2 --output_path npz_data

# 2. Training
python train.py

# 3. Inference + Grad-CAM (exactly as in Figure 1)
python inference.py --input sample.jpg --weights weights/best_model.pth --output result.jpg

# ForensicFlow
ForensicFlow: A Tri-Modal Adaptive Network for Robust Deepfake Detection (arXiv 2025) AUC 0.9752 • Full-frame processing • No face cropping • Preserves forensic artifacts RGB + Texture (Swin-T) + Frequency branches • Attention-based temporal fusion Official code + weights + Grad-CAM 
