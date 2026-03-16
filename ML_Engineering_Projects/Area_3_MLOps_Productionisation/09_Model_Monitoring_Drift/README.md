# Model Monitoring & Drift Detection

Post-deployment ML monitoring system with statistical drift detection (KS test, PSI, chi-squared, MMD), prediction distribution tracking, threshold-based alerting, and a Streamlit dashboard.

---

## Drift Types Covered

| Type | What changes | Detection method |
|------|-------------|-----------------|
| **Data drift** | Input text distribution shifts | KS test, PSI on features |
| **Concept drift** | Relationship between input and label changes | Prediction distribution shift |
| **Prediction drift** | Model starts predicting different labels | Chi-squared on label counts |
| **Embedding drift** | Semantic content shifts | MMD on text embeddings |

---

## Architecture

```
STEP 1: Profile Reference (run once on training/known-good data)
──────────────────────────────────────────────────────────────
Reference Data (IMDB test set)
    │
    ▼
┌──────────────────────┐
│ ReferenceProfiler    │  compute mean, std, quantiles, label dist
└──────────┬───────────┘
           │
           ▼
    reference_profile.json

STEP 2: Monitor Production (run periodically on new batches)
──────────────────────────────────────────────────────────────
New Data (Yelp reviews = domain drift)
    │
    ▼
┌──────────────────────┐    ┌──────────────────────┐
│ DriftDetector        │    │ PredictionMonitor     │
│ ├── KS test          │    │ ├── rolling label dist │
│ ├── PSI              │    │ └── entropy tracking  │
│ ├── chi-squared      │    └──────────┬────────────┘
│ └── MMD (embeddings) │               │
└──────────┬───────────┘               │
           └──────────┬────────────────┘
                      ▼
              ┌───────────────┐
              │    Alerter    │  threshold check → log/email/Slack
              └───────┬───────┘
                      │
                      ▼
              ┌───────────────┐
              │   Dashboard   │  Streamlit + Plotly interactive UI
              └───────────────┘
```

---

## Dataset

| Dataset | Use |
|---------|-----|
| IMDB | Reference distribution (training domain) |
| Yelp Reviews | Production simulation (domain shift) |
| Amazon Reviews | Temporal drift simulation |

See [data.txt](data.txt) for download links and the recommended drift simulation workflow.

---

## Setup

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
```

---

## Usage

### Step 1: Profile reference data (run once)

```bash
uv run python scripts/profile_reference.py \
    --dataset imdb \
    --split test \
    --output data/reference_profile.json
```

### Step 2: Monitor new batch

```bash
uv run python scripts/monitor_batch.py \
    --dataset yelp_review_full \
    --split test \
    --reference data/reference_profile.json \
    --output data/drift_results/
```

### Step 3: View dashboard

```bash
uv run python scripts/run_dashboard.py
# Opens browser at http://localhost:8501
```

### Run tests

```bash
uv run pytest
```

---

## Example Dashboard

```
┌─────────────────────────────────────────────────────────────┐
│  ML Monitoring Dashboard                    [Date: 2024-01] │
├─────────────────────────────────────────────────────────────┤
│  Feature Drift                                              │
│  text_length    PSI=0.31  [ALERT] ██████████░░░░░           │
│  avg_word_len   PSI=0.08  [OK]    ██░░░░░░░░░░░░░           │
│                                                             │
│  Prediction Distribution                                    │
│  Reference:  NEGATIVE=52%  POSITIVE=48%                     │
│  Current:    NEGATIVE=71%  POSITIVE=29%  [DRIFT DETECTED]   │
│                                                             │
│  Alerts (last 24h)                                          │
│  [WARN] 2024-01-15 09:32  PSI > 0.25 on text_length        │
│  [WARN] 2024-01-15 09:32  Prediction distribution shifted  │
└─────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
09_Model_Monitoring_Drift/
├── pyproject.toml
├── .python-version
├── README.md
├── data.txt
├── src/
│   ├── reference_profiler.py  # stats from training data
│   ├── drift_detector.py      # KS, PSI, chi-squared, MMD + Evidently
│   ├── prediction_monitor.py  # rolling prediction distribution
│   ├── data_quality.py        # missing values, length outliers
│   ├── alerter.py             # threshold rules + dispatch
│   └── dashboard.py           # Streamlit app
├── scripts/
│   ├── profile_reference.py   # CLI: build reference profile
│   ├── monitor_batch.py       # CLI: run drift detection on new data
│   └── run_dashboard.py       # CLI: launch Streamlit dashboard
├── notebooks/
│   ├── 01_simulate_drift.ipynb
│   └── 02_drift_analysis.ipynb
├── tests/
│   ├── test_drift_detector.py
│   └── test_prediction_monitor.py
└── data/
    ├── reference_profile.json  # created by profile_reference.py
    └── drift_results/          # created by monitor_batch.py
```

---

## Alert Thresholds (configurable)

| Test | Threshold | Action |
|------|-----------|--------|
| PSI > 0.1 | Warning | Log |
| PSI > 0.25 | Alert | Log + notify |
| KS p-value < 0.05 | Alert | Log + notify |
| Prediction entropy drop > 50% | Alert | Log + notify |

---

## Integration with Other Projects

- Connects to **Project 07** (REST API) to monitor production predictions
- Connects to **Project 08** (Batch Pipeline) to monitor batch output Parquet files
- Reference data = Project 04's training set distribution

---

## Future Improvements

- Real-time monitoring via Kafka/Redis queue
- Automated retraining trigger when drift is detected
- SHAP-based feature importance drift
