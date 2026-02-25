# 🚗 Sensor Fault Detection in Automotive IMU Sensors

> **Machine Learning-based fault classification for MPU-9250 IMU sensors using the Passive Vehicular Sensors (PVS) dataset**

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3+-orange?logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)
![Dataset](https://img.shields.io/badge/Dataset-PVS%20(Kaggle)-lightblue)

---

## 📋 Table of Contents
- [Overview](#overview)
- [Results](#results)
- [Project Structure](#project-structure)
- [Pipeline](#pipeline)
- [Dataset](#dataset)
- [Fault Types](#fault-types)
- [Features](#features)
- [Models](#models)
- [How to Run](#how-to-run)
- [Team](#team)
- [References](#references)

---

## Overview

Modern vehicles rely on Inertial Measurement Units (IMUs) for stability control, ADAS, and autonomous driving. A faulty IMU sensor silently propagates wrong data into braking, steering, and lane-keeping systems — with potentially dangerous consequences.

This project builds a complete **supervised machine learning pipeline** that:
- Automatically classifies 4 states: `normal`, `noise_fault`, `bias_fault`, `flatline_fault`
- Operates on real-world driving data from the **PVS dataset** (9 contexts, 3 vehicles, 3 drivers)
- Uses **synthetic fault injection** calibrated to actual signal statistics
- Achieves **90.78% accuracy** with an SVM (RBF) classifier

---

## Results

| Model | Accuracy | F1 Macro | Macro AUC | Inference |
|---|---|---|---|---|
| **SVM (RBF)** ⭐ | **90.78%** | **90.81%** | **0.9845** | 0.648 ms/sample |
| XGBoost | 90.71% | 90.72% | — | 0.023 ms/sample |
| Random Forest | 90.51% | 90.55% | — | 0.038 ms/sample |
| KNN (k=7) | 88.85% | 88.91% | — | 0.076 ms/sample |

### Per-Class Performance (Best Model — SVM RBF)

| Class | Precision | Recall | F1 | Notes |
|---|---|---|---|---|
| normal | 78.76% | 89.21% | 83.66% | Hardest boundary with noise_fault |
| noise_fault | 89.74% | 80.27% | 84.74% | 18.4% confused with normal |
| bias_fault | 96.07% | 93.64% | 94.84% | Trend feature highly effective |
| **flatline_fault** | **100%** | **100%** | **100%** | Perfect — unique_ratio is definitive |

> 💡 **XGBoost** achieves near-identical accuracy to SVM while being **28× faster** at inference — recommended for real-time deployment.

---

## Project Structure

```
📦 sensor-fault-detection-imu/
│
├── 📓 notebooks/
│   ├── Phase1_Temperature_EDA.ipynb              # Exploratory data analysis
│   ├── Phase2_Fault_Injection_Enhanced.ipynb     # Calibrated fault injection
│   ├── Phase3_Feature_Extraction.ipynb           # Feature engineering
│   ├── Phase4_Model_Training.ipynb               # Train & compare models
│   └── Phase5_Evaluation_Report.ipynb            # Full evaluation metrics
│
├── 🐍 scripts/
│   └── generate_report.py                        # PDF report generator
│
├── 📄 docs/
│   └── Project_Report_Final.pdf                  # Full project report (professor submission)
│
├── 📊 results/
│   └── sample_plots/                             # Example output plots
│
├── requirements.txt                              # Python dependencies
├── .gitignore                                    # Files excluded from git
└── README.md                                     # This file
```

---

## Pipeline

```
Raw PVS CSV Files
      │
      ▼
┌─────────────────┐
│   Phase 1: EDA  │  ── Signal statistics, fault parameter calibration
└────────┬────────┘
         │
         ▼
┌──────────────────────┐
│  Phase 2: Fault      │  ── Inject noise/bias/flatline into windows
│  Injection           │     Verify difficulty (target: 80–95% F1)
└────────┬─────────────┘
         │
         ▼
┌──────────────────────┐
│  Phase 3: Feature    │  ── 8 time-domain features per 100-sample window
│  Extraction          │     Mutual Information ranking, correlation filter
└────────┬─────────────┘
         │
         ▼
┌──────────────────────┐
│  Phase 4: Model      │  ── RF, SVM, KNN, XGBoost
│  Training            │     5-fold CV, stratified 80/20 split
└────────┬─────────────┘
         │
         ▼
┌──────────────────────┐
│  Phase 5: Evaluation │  ── Confusion matrix, ROC curves, per-class metrics
│  Report              │     PDF report generation
└──────────────────────┘
```

---

## Dataset

**Passive Vehicular Sensors (PVS) Dataset**
- 📍 Collected in Brazil by Menegazzo & von Wangenheim (UFSC)
- 🚗 3 vehicles: VW Saveiro, Fiat Bravo, Fiat Palio
- 👤 3 drivers, 3 road scenarios = **9 dataset combinations**
- 📡 Sensor: MPU-9250 (Accelerometer, Gyroscope, Magnetometer, Temperature) at **100 Hz**
- 📦 [Download from Kaggle](https://www.kaggle.com/datasets/jmenegazzo/passive-vehicular-sensors-dataset)

After downloading, place the folders as:
```
C:\PVS_Dataset\
├── PVS1\
│   ├── dataset_mpu_left.csv
│   └── dataset_mpu_right.csv
├── PVS2\
│   └── ...
...
└── PVS9\
```

> Change `BASE_PATH` in each notebook Cell 1 to match your local path.

---

## Fault Types

| Fault | Label | Description | Injection Method |
|---|---|---|---|
| Normal | 0 | Clean sensor output | Raw signal + ADC noise (σ × 0.15) |
| Noise/Spike | 1 | Random amplitude spikes | 3–7 spikes per window, amplitude 1.5σ–4.0σ |
| Bias/Drift | 2 | Gradual linear offset | Drift 0.8σ–2.5σ starting at random point |
| Flatline | 3 | Sensor stuck at constant | Window mean + jitter ≈ 0 |

Fault amplitudes are calibrated to the **global standard deviation** of the actual signal to ensure realistic, non-trivial detection difficulty. A quick-check RF is run after injection — target F1 is 80–95%.

---

## Features

8 time-domain features extracted per 100-sample window:

| Feature | What it detects |
|---|---|
| `mean` | Shifts under bias/drift |
| `std` | Near-zero in flatline; elevated in noise |
| `min` | Negative spikes |
| `max` | Positive spikes |
| `skewness` | Asymmetry from one-sided spikes |
| `kurtosis` | Heavy tails — impulsive noise |
| `trend` | Linear slope — non-zero in bias fault |
| `unique_ratio` | Fraction of unique values — ≈0 in flatline |

Features selected after removing highly correlated pairs (|r| > 0.90) and ranking by Mutual Information.

---

## Models

```python
# Models trained and compared
models = {
    'Random Forest' : RandomForestClassifier(n_estimators=200, class_weight='balanced'),
    'KNN'           : KNeighborsClassifier(n_neighbors=7, weights='distance'),
    'SVM'           : SVC(C=10, kernel='rbf', gamma='scale', class_weight='balanced'),
    'XGBoost'       : XGBClassifier(n_estimators=300, learning_rate=0.1, max_depth=6),
}
```

---

## How to Run

### Option A — Local (VS Code / Jupyter)

**1. Clone the repo**
```bash
git clone https://github.com/YOUR_USERNAME/sensor-fault-detection-imu.git
cd sensor-fault-detection-imu
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Download the dataset**

Download PVS dataset from [Kaggle](https://www.kaggle.com/datasets/jmenegazzo/passive-vehicular-sensors-dataset) and set `BASE_PATH` in each notebook.

**4. Run notebooks in order**
```
Phase1 → Phase2 → Phase3 → Phase4 → Phase5
```

---

### Option B — Google Colab

Click the badge below to open each notebook directly in Colab:

| Notebook | Colab Link |
|---|---|
| Phase 1 — EDA | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/sensor-fault-detection-imu/blob/main/notebooks/Phase1_Temperature_EDA.ipynb) |
| Phase 2 — Fault Injection | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/sensor-fault-detection-imu/blob/main/notebooks/Phase2_Fault_Injection_Enhanced.ipynb) |
| Phase 3 — Features | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/sensor-fault-detection-imu/blob/main/notebooks/Phase3_Feature_Extraction.ipynb) |
| Phase 4 — Training | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/sensor-fault-detection-imu/blob/main/notebooks/Phase4_Model_Training.ipynb) |
| Phase 5 — Evaluation | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/sensor-fault-detection-imu/blob/main/notebooks/Phase5_Evaluation_Report.ipynb) |

> Replace `YOUR_USERNAME` with your actual GitHub username after uploading.

---

## Team

| Name | Roll Number |
|---|---|
| NCSS Uday | SM25mtech11002 |
| Shashi Prakash | SM25mtech11008 |
| Bavan Prabahar | SM25mtech11009 |

**Course:** Automotive Communication and Sensing
**Institution:** NIT / MTECH Program

---

## References

1. J. Menegazzo and A. von Wangenheim, *"Multi-Contextual and Multi-Aspect Analysis for Road Surface Type Classification Through Inertial Sensors and Deep Learning,"* SBESC 2020. DOI: [10.1109/SBESC51047.2020.9277846](https://doi.org/10.1109/SBESC51047.2020.9277846)

2. J. Menegazzo and A. von Wangenheim, *"Road Surface Type Classification Based on Inertial Sensors and Machine Learning: A Systematic Review,"* SN Computer Science, 2021. DOI: [10.1007/s42979-021-00901-2](https://doi.org/10.1007/s42979-021-00901-2)

3. S. Safavi et al., *"Multi-Sensor Fault Detection, Identification, Isolation and Health Forecasting for Autonomous Vehicles,"* Sensors, 2021. DOI: [10.3390/s21072547](https://doi.org/10.3390/s21072547)

4. PVS Dataset on Kaggle: [https://www.kaggle.com/datasets/jmenegazzo/passive-vehicular-sensors-dataset](https://www.kaggle.com/datasets/jmenegazzo/passive-vehicular-sensors-dataset)

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
