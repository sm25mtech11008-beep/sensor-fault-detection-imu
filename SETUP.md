# Setup Guide

## Prerequisites

- Python 3.10 or higher
- ~4 GB free disk space (for dataset)
- 8 GB RAM recommended

---

## Step 1 — Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/sensor-fault-detection-imu.git
cd sensor-fault-detection-imu
```

---

## Step 2 — Install Dependencies

```bash
pip install -r requirements.txt
```

To verify everything installed correctly:
```bash
python -c "import sklearn, xgboost, pandas, reportlab; print('All packages OK')"
```

---

## Step 3 — Download the Dataset

1. Go to [Kaggle PVS Dataset](https://www.kaggle.com/datasets/jmenegazzo/passive-vehicular-sensors-dataset)
2. Download and extract
3. Arrange folders as follows:

```
C:\PVS_Dataset\          ← Windows
/home/user/PVS_Dataset/  ← Linux / Mac / Colab
│
├── PVS1\
│   ├── dataset_mpu_left.csv
│   └── dataset_mpu_right.csv
├── PVS2\
│   ├── dataset_mpu_left.csv
│   └── dataset_mpu_right.csv
...
└── PVS9\
    ├── dataset_mpu_left.csv
    └── dataset_mpu_right.csv
```

---

## Step 4 — Configure Base Path

In **every notebook**, Cell 1 contains:

```python
BASE_PATH = r'C:\PVS_Dataset'   # Windows default
```

Change this to match your local path:

```python
# Linux / Mac
BASE_PATH = '/home/yourname/PVS_Dataset'

# Google Colab (after uploading to Drive)
BASE_PATH = '/content/drive/MyDrive/PVS_Dataset'
```

---

## Step 5 — Run Notebooks in Order

```
Phase1_Temperature_EDA.ipynb
        ↓  (generates temp_config.json)
Phase2_Fault_Injection_Enhanced.ipynb
        ↓  (generates temp_labeled_dataset.parquet)
Phase3_Feature_Extraction.ipynb
        ↓  (generates temp_splits.pkl, temp_scaler.pkl)
Phase4_Model_Training.ipynb
        ↓  (generates all_models.pkl)
Phase5_Evaluation_Report.ipynb
        ↓  (generates all evaluation plots)
scripts/generate_report.py
        ↓  (generates Fault_Detection_Report.pdf)
```

Each phase reads config files from the previous phase automatically.
Do **not** skip phases or run them out of order.

---

## Running on Google Colab

1. Upload the repo to your Google Drive
2. Mount Drive in Colab:
```python
from google.colab import drive
drive.mount('/content/drive')
```
3. Set `BASE_PATH = '/content/drive/MyDrive/PVS_Dataset'` in Cell 1 of each notebook
4. Run `!pip install -r requirements.txt` in a first cell

---

## Output Folders Created Automatically

| Folder | Contents |
|---|---|
| `outputs_temp_phase1/` | EDA plots, signal statistics, config JSON |
| `outputs_temp_phase2/` | Labeled dataset parquet, fault injection config |
| `outputs_temp_phase3/` | Feature matrix, train/test splits, scaler |
| `outputs_temp_phase4/` | Trained models, confusion matrices, comparison plots |
| `outputs_temp_phase5/` | ROC curves, per-class metrics, evaluation plots |
| `report_plots/` | All plots used in PDF report |

All folders are created inside `BASE_PATH` automatically.

---

## Common Issues

**`ModuleNotFoundError: No module named 'xgboost'`**
```bash
pip install xgboost
```

**`FileNotFoundError: temp_phase2_config.json not found`**
→ You skipped Phase 2. Run phases in order.

**`MemoryError` when loading CSVs**
→ The notebooks load only 1 column at a time (`usecols=[TEMP_COL]`). If still failing, reduce `CHUNK_SIZE` in Cell 2 from 50,000 to 20,000.

**SVM running very slowly**
→ Normal. The SVM cell auto-subsamples to 80,000 windows if the dataset is larger.
