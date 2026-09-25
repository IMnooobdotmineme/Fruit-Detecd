# Fruit & Vegetable Freshness Classifier (Fruit-Detecd)

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-MobileNetV2-D00000?style=flat&logo=keras&logoColor=white)](https://keras.io/)


An automated Computer Vision (CNN) deep learning classifier that detects freshness and staleness in fruits and vegetables using Transfer Learning with MobileNetV2.

---

## Project Overview
* **Author:** Ro Arthiphu
* **Category:** Computer Vision (CNN)
* **Objective:** Classify fruits and vegetables as fresh or stale to support automated food quality assessment.
* **Architecture:** MobileNetV2 pretrained on ImageNet with customized classification head and data augmentation.

---

## Performance & Results
The model was trained in two distinct phases (Feature Extraction & Fine-Tuning) using Google Colab T4 GPU acceleration:

| Phase | Epochs | Learning Rate | Train Accuracy | Validation Accuracy | Validation Loss |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Phase 1: Feature Extraction** | 5 | $1 \times 10^{-3}$ | 97.24% | 97.56% | 0.0686 |
| **Phase 2: Fine-Tuning** | 5 | $1 \times 10^{-5}$ | **98.13%** | **98.54%** | **0.0442** |

---

## Dataset
* **Source:** Kaggle (`swoyam2609/fresh-and-stale-classification`)
* **Total Samples:** ~21,513 images (16,023 training / 5,490 validation)
* **Classes:** Synchronized 10-class dataset covering fresh and stale varieties of apples, bananas, oranges, and vegetables.

---

## Repository Structure
```text
Fruit-Detecd/
│
├── download_data.py    # Dataset downloading script via kagglehub
├── train.py            # Complete training pipeline (Augmentation, MobileNetV2, Fine-Tuning)
├── fruit_model.h5      # Pre-trained MobileNetV2 weights (HDF5 format)
└── README.md           # Project documentation
