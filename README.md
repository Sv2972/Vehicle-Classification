# Vehicle Classification System (12-Class)

## Project Overview
This repository contains a robust deep learning pipeline for classifying 12 types of vehicles from images, including specialized categories like auto-rickshaws, e-rickshaws, and mini-trucks. The system handles data imbalance and noise through automated auditing and weighted sampling.

## Key Results
* **Training Accuracy**: 92.58%
* **Validation Accuracy**: 85.00%
* **Architecture**: EfficientNet-B0 (Pre-trained on ImageNet-1K)
* **Deployment Format**: ONNX (Verified)

## Repository Structure
- `data/`: Contains raw and cleaned datasets.
- `src/data_prep.py`: Script for Cleanvision-based data auditing and cleaning.
- `src/train.py`: Main training script with imbalance handling and ONNX export.
- `src/verify_onnx.py`: Utility to verify the exported ONNX model.
- `models/`: Stores the exported `.onnx` model and training `learning_curves.png`.
- `classes.txt`: Ordered list of the 12 vehicle classes.

## How to Run
1. **Data Preparation**: `python src/data_prep.py`
2. **Training**: `python src/train.py`
3. **Verification**: `python src/verify_onnx.py`