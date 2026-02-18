# 🔍 Financial Fraud Detection — Anomaly Analysis

> **Unsupervised anomaly detection on financial transaction data using Isolation Forest, with full model evaluation, threshold optimisation, and business impact analysis.**

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Key Results](#key-results)
- [Model Evaluation](#model-evaluation)
- [Quick Start](#quick-start)
- [Notebook Guide](#notebook-guide)
- [Outputs](#outputs)
- [Actionable Recommendations](#actionable-recommendations)
- [Future Work](#future-work)

---

## Project Overview

This project detects potentially fraudulent financial transactions using **unsupervised machine learning** — no labelled fraud data required. An **Isolation Forest** model assigns every transaction a continuous anomaly score between 0 (normal) and 1 (highly anomalous), then flags the top 5% as suspicious.

The analysis is split into two parts:

| Part | Sections | Focus |
|---|---|---|
| **I — Detection** | 1 – 8 | EDA, feature engineering, model training, visualisations, findings |
| **II — Evaluation** | 9 – 13 | Statistical validation, threshold optimisation, robustness, business impact |

---

## Repository Structure

```
fraud-detection-repo/
│
├── notebooks/
│   └── fraud_anomaly_detection.ipynb   # Full analysis (48 cells, 13 sections)
│
├── data/
│   └── transactions.csv                # Raw transaction dataset (2,512 rows)
│
├── docs/
│   ├── executive_summary.docx
│   └── executive_summary.pptx          # Non-technical stakeholder report
│
├── outputs/                            # Generated charts (populated on run)
│   ├── anomalies_summary.csv
│   ├── anomalies_histogram.png
│   ├── roc_pr_curves.png
│   ├── threshold_optimisation.png
│   ├── bootstrap_stability.png
│   ├── robustness_sensitivity.png
│   ├── business_impact.png
│   └── evaluation_dashboard.png
│
├── requirements.txt                    # Python dependencies
└── README.md                           # This file
```

---

## Dataset

**File:** `data/transactions.csv`  
**Rows:** 2,512 transactions  
**Features:** 16 columns

| Column | Type | Description |
|---|---|---|
| `TransactionID` | str | Unique transaction identifier |
| `AccountID` | str | Customer account identifier |
| `TransactionAmount` | float | Amount in USD |
| `TransactionDate` | datetime | Timestamp of transaction |
| `TransactionType` | str | Debit / Credit |
| `Location` | str | City of transaction |
| `DeviceID` | str | Device used |
| `IP Address` | str | Originating IP |
| `MerchantID` | str | Merchant identifier |
| `Channel` | str | ATM / Online / Branch |
| `CustomerAge` | int | Customer age |
| `CustomerOccupation` | str | Occupation category |
| `TransactionDuration` | int | Session duration (seconds) |
| `LoginAttempts` | int | Number of login attempts |
| `AccountBalance` | float | Balance at time of transaction |
| `PreviousTransactionDate` | datetime | Date of prior transaction |

---

## Methodology

### Feature Engineering

Three fraud-signal features are derived before modelling:

| Feature | Formula | Fraud Signal |
|---|---|---|
| `Amount_to_Balance_Ratio` | `TransactionAmount / AccountBalance` | Spending > balance is strongly anomalous |
| `Hour` | Hour extracted from `TransactionDate` | Off-hours transactions are riskier |
| `HighLoginAttempts` | `LoginAttempts > 1` | Multiple attempts → credential abuse |

### Model: Isolation Forest

**Why Isolation Forest?**  
Fraudulent transactions are rare and structurally different from normal ones. Isolation Forest randomly partitions the feature space; anomalous points require fewer splits to isolate — making it naturally suited to imbalanced fraud data without needing labels.

**Hyperparameters:**

```python
IsolationForest(
    n_estimators   = 200,     # 200 trees for stable scoring
    contamination  = 0.05,    # Assume ~5% fraud rate
    random_state   = 42,      # Reproducible results
    max_features   = 1.0      # All features used per split
)
```

**Feature matrix used for training:**

```python
features = [
    'TransactionAmount',
    'TransactionDuration',
    'LoginAttempts',
    'AccountBalance',
    'Amount_to_Balance_Ratio',
    'Hour'
]
```

---

## Key Results

| Metric | Value |
|---|---|
| Total transactions | 2,512 |
| Flagged anomalies | **126 (5.02%)** |
| Mean amount — normal | $277 |
| Mean amount — anomalous | **$686 (+148%)** |
| Mean login attempts — normal | 1.04 |
| Mean login attempts — anomalous | **2.68 (+158%)** |
| Mean Amount/Balance ratio — normal | 0.15 |
| Mean Amount/Balance ratio — anomalous | **1.17 (+680%)** |
| Top anomaly score | **1.000** (TX000275) |

### Top 5 Highest-Risk Transactions

| TransactionID | Amount | Login Attempts | Balance | Ratio | Score |
|---|---|---|---|---|---|
| TX000275 | $1,176 | 5 | $324 | 3.63× | 1.000 |
| TX000899 | $1,531 | 4 | $860 | 1.78× | 0.999 |
| TX000455 | $611 | 4 | $920 | 0.66× | 0.913 |
| TX000615 | $1,342 | 1 | $694 | 1.93× | 0.900 |
| TX000148 | $515 | 5 | $422 | 1.22× | 0.883 |

---

## Model Evaluation

Full evaluation is covered in **Sections 9–13** of the notebook. Since no ground-truth labels are available, a **Local Outlier Factor (LOF)** model is used as an independent pseudo-ground-truth.

### Statistical Validation

| Metric | Value | Interpretation |
|---|---|---|
| ROC-AUC (vs LOF) | — | Run notebook to compute |
| PR-AUC (vs LOF) | — | Run notebook to compute |
| Silhouette Score | — | Run notebook to compute |
| Detector Agreement | — | Run notebook to compute |
| Score Correlation (ISO ↔ LOF) | — | Run notebook to compute |

> Metrics are computed at runtime — values are dataset-specific and printed in Section 9.

### Threshold Operating Tiers

| Tier | Score Range | Action | Est. Volume |
|---|---|---|---|
| ✅ Auto-approve | < 0.45 | Process normally | ~90% |
| ⚠️ Soft flag | 0.45 – 0.75 | Step-up authentication | ~5% |
| 🚫 Hard block | ≥ 0.75 | Human review queue | ~5% |

### Robustness

- **Bootstrap stability** — 50 runs with different seeds; mean score variance < 0.05 indicates reliable flagging
- **Contamination sweep** — tested at 2%, 5%, 8%, 10%; transactions flagged at all levels = highest-confidence fraud
- **Feature permutation importance** — `Amount_to_Balance_Ratio` is the most discriminating feature

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/your-org/fraud-detection-repo.git
cd fraud-detection-repo
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Launch the notebook

```bash
jupyter lab notebooks/fraud_anomaly_detection.ipynb
```

> **Note:** Place `transactions.csv` in the `data/` directory before running. Update the file path in Cell 4 if needed:
> ```python
> transactions = pd.read_csv('../data/transactions.csv')
> ```

### 5. Run all cells

Use **Kernel → Restart & Run All** to execute the full pipeline end-to-end.

---

## Notebook Guide

| Section | Cells | What it does |
|---|---|---|
| **1 — Setup** | 2 | Import libraries, configure plot style |
| **2 — EDA** | 3 | Distribution plots, categorical breakdowns |
| **3 — Feature Engineering** | 2 | Derive ratio, hour, login flag; correlation heatmap |
| **4 — Model Training** | 1 | Train Isolation Forest, compute `Anomaly_Score`, set `Anomaly` flag |
| **5 — Anomalies Summary** | 3 | Build `anomalies_summary` DataFrame, comparison stats, top 15 |
| **6 — Visualisations** | 5 | Histogram, score dist., scatter, channel/hour heatmap, login breakdown |
| **7 — Findings** | 2 | Written findings + quantified evidence |
| **8 — Recommendations** | 2 | Actionable security, trust & efficiency insights; export CSV |
| **9 — Statistical Validation** | 4 | LOF comparison, ROC/PR curves, silhouette, confusion matrix |
| **10 — Threshold Optimisation** | 3 | F1/F2/Youden sweep, three-tier system |
| **11 — Robustness** | 4 | Bootstrap stability, contamination sweep, permutation importance |
| **12 — Business Impact** | 2 | Cost/benefit analysis, net value curve |
| **13 — Dashboard** | 2 | Full evaluation scorecard, final summary print |

---

## Outputs

All outputs are saved to the `outputs/` directory when the notebook is run:

| File | Description |
|---|---|
| `anomalies_summary.csv` | 126-row DataFrame of flagged transactions |
| `anomalies_histogram.png` | Amount distribution: normal vs anomalous |
| `eda_distributions.png` | Key numeric feature distributions |
| `eda_categorical.png` | Categorical breakdowns |
| `correlation_heatmap.png` | Feature correlation matrix |
| `score_distribution.png` | Anomaly score distribution |
| `scatter_amount_balance.png` | Amount vs balance scatter (score-coloured) |
| `anomaly_patterns.png` | Anomaly rate by channel & hour |
| `login_attempts.png` | Login attempt breakdown |
| `roc_pr_curves.png` | ROC and PR curves vs LOF pseudo-labels |
| `confusion_agreement.png` | ISO vs LOF agreement matrix |
| `threshold_optimisation.png` | P/R/F1/F2 curves + volume vs threshold |
| `bootstrap_stability.png` | Score variance across 50 bootstrap runs |
| `robustness_sensitivity.png` | Contamination sweep + feature importance |
| `business_impact.png` | Net value curve + cost breakdown |
| `evaluation_dashboard.png` | Full evaluation scorecard |

---

## Actionable Recommendations

### 🔒 Security
1. **Hard block** transactions where `Amount_to_Balance_Ratio > 1.0` — require biometric/OTP confirmation
2. **Throttle login attempts** — CAPTCHA after 2 failures; flag any subsequent transaction from that session
3. **Velocity caps** — block transactions exceeding 3× the account's 30-day mean amount without out-of-band confirmation

### 🛡️ Trust
4. **Graduated friction** — score < 0.45 auto-approve; 0.45–0.75 re-authenticate; > 0.75 hard block
5. **Transparent alerts** — instant push notification with plain-language reason and one-tap dispute flow
6. **Account health dashboard** — customer-facing anomaly indicator builds long-term trust

### ⚙️ Efficiency
7. **Prioritised review queue** — sort flagged transactions by `Anomaly_Score` descending; top 20 first
8. **Enrich features** — add device fingerprint consistency, geo-velocity, and merchant category codes
9. **Monthly retraining** — schedule on newly labelled data; monitor score-distribution drift as KPI
10. **Automate tiers** — three-tier thresholds reduce analyst workload by an estimated 60–70%

---

## Future Work

- [ ] Collect confirmed fraud labels to enable supervised model training (XGBoost, LightGBM)
- [ ] Add real-time scoring API endpoint (FastAPI / Flask)
- [ ] Integrate device fingerprint and geo-velocity features
- [ ] Build a live monitoring dashboard (Grafana / Streamlit)
- [ ] Implement model drift detection with automated retraining trigger
- [ ] A/B test three-tier threshold system against current rule-based system

---

## License

This project is for internal analysis purposes. Data is confidential and must not be shared externally.

---

*Analysis produced using Python 3.10 · scikit-learn · pandas · matplotlib*
