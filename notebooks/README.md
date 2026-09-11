# Notebooks — the original (v1) pipeline

These five notebooks are the **original class-project submission**, kept as the
exploratory record: EDA, calibration decisions, and the plots that went into
`docs/Project_Report_Final.pdf`.

They are **not** the current pipeline. The logic they contain has been moved
into the tested `sfd` package under [`../src/sfd/`](../src/sfd/), and three
methodological problems found while doing so are documented in the root
[README](../README.md#what-changed-in-v2):

| Notebook | Kept for | Superseded by |
|---|---|---|
| `Phase1_Temperature_EDA.ipynb` | Signal statistics, left/right comparison | `sfd.data.signal_statistics` |
| `Phase2_Fault_Injection_Enhanced.ipynb` | Fault calibration reasoning | `sfd.faults` |
| `Phase3_Feature_Extraction.ipynb` | Feature motivation, MI ranking | `sfd.features` |
| `Phase4_Model_Training.ipynb` | Model comparison | `sfd.models` |
| `Phase5_Evaluation_Report.ipynb` | Confusion matrices, ROC | `sfd.evaluate` |

Each notebook still hardcodes `BASE_PATH = r'C:\PVS_Dataset'` in cell 1. The
package reads `PVS_DATA` from the environment instead — see
[`sfd/config.py`](../src/sfd/config.py).

To run them, install the notebook extras:

```bash
pip install -e ".[notebooks]"
```
