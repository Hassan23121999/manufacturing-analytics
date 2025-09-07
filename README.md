# ⚙️ Manufacturing Analytics with PySpark

This project demonstrates a **high-level ETL + ML pipeline** for manufacturing data (parts, production events, quality checks, maintenance logs).  
It is designed as a **prototype for industrial analytics** and as an **demo** of PySpark, FastAPI, and Streamlit.

---

##  Project Overview
- **Bronze Layer** → Ingest raw CSVs into Parquet (schema captured, raw preserved).  
- **Silver Layer** → Clean & standardize data (null handling, type casting, outlier removal).  
- **Gold Layer** → Business-ready features:
  - Machine KPIs (cycle time, scrap rate)
  - Maintenance KPIs (downtime, costs)
  - Per-part feature views (geometry, QC, energy, scrap rate)
- **ML Pipeline** → Train + Score a cost prediction model on part features.  
- **API (FastAPI)** → Trigger ETL/ML jobs.  
- **UI (Streamlit)** → Upload CSVs, run pipeline, view KPIs, visualize predictions, export results.

---

##  Project Structure
```
manufacturing-analytics/
│
├── data/                # Raw input CSVs (parts, production, quality, maintenance)
├── bronze/              # Bronze layer parquet outputs
├── silver/              # Silver layer cleaned parquet outputs
├── gold/                # Gold layer features & KPIs
│   ├── kpi_machine/
│   ├── kpi_maintenance/
│   ├── features_per_part/
│   └── predictions/
│
├── jobs/                # PySpark ETL/ML jobs
│   ├── ingest_parts.py
│   ├── ingest_production.py
│   ├── ingest_maintenance.py
│   ├── silver_transform.py
│   ├── gold_features.py
│   ├── ml_train.py
│   └── ml_score.py
│
├── models/              # Saved ML models
├── api/                 # FastAPI backend
│   └── main.py
├── ui/                  # Streamlit dashboard
│   └── dashboard.py
├── scripts/             # Helpers (e.g. synthetic data generator)
└── README.md            # This file
```

---

##  Workflow

1. **Generate synthetic data**  
   ```bash
   python scripts/generate_synthetic.py --parts 1000 --machines 10 --events-min 5 --events-max 10 --insp-min 1 --insp-max 3 --format csv
   ```

2. **Run API (triggers pipeline jobs)**
   ```bash
   uvicorn api.main:app --reload
   ```

3. **Run UI (interactive dashboard)**
   ```bash
   streamlit run ui/dashboard.py
   ```

4. **Upload data** (`parts.csv`, `production.csv`, `quality.csv`, `maintenance.csv`) in the sidebar.
   Then click **"Run Full Pipeline 🚀"** to execute ETL + ML.

---

##  Dashboard Features

- **Machines tab** → Scrap rate & cycle time per machine (bar charts, filterable).
- **Parts tab** → Per-part features with material filter + scatter plot of complexity vs cycle time.
- **Predictions tab** → Predicted manufacturing cost distribution with cost filter + histogram.
- **Export** → Download any table as CSV/Excel for reporting.

---

## 🧑 Tech Stack

- **PySpark** — ETL, aggregations, MLlib regression model.
- **FastAPI** — REST endpoints to run jobs.
- **Streamlit** — UI for upload, pipeline execution, KPIs & visualizations.
- **Parquet** — Efficient columnar storage for Bronze/Silver/Gold layers.

---

##  Interview Talking Points

- Demonstrates **Data Lakehouse design (Bronze/Silver/Gold)**.
- Shows **data cleaning (nulls, imputation, outliers)**.
- Builds **KPIs relevant to manufacturing**.
- Trains + serves a **Spark ML pipeline**.
- Wraps with **API + UI** → looks like a real industrial solution.

---

##  Cleanup

Stop Spark sessions and servers when done:
```bash
# Stop Streamlit (Ctrl+C)
# Stop FastAPI (Ctrl+C)
```

---

##  Next Ideas

- Add anomaly detection for QC data.
- Connect to real shop-floor IoT streams (Kafka).
- Deploy on **Databricks** or **AWS EMR** for scale.