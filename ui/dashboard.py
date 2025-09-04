import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from io import BytesIO

API = "http://localhost:8000"
DATA_DIR = Path("data")

st.set_page_config(page_title="Manufacturing Analytics", layout="wide")
st.title("⚙️ Manufacturing Analytics Dashboard")

# -------------------------
# Sidebar: Upload data
# -------------------------
st.sidebar.header("Upload Raw CSVs")

def save_uploaded_file(upload, name):
    DATA_DIR.mkdir(exist_ok=True)
    out_path = DATA_DIR / name
    with open(out_path, "wb") as f:
        f.write(upload.getbuffer())
    st.sidebar.success(f"Saved {name}")
    return out_path

for fname in ["parts.csv", "production.csv", "quality.csv", "maintenance.csv"]:
    upload = st.sidebar.file_uploader(f"Upload {fname}", type="csv")
    if upload:
        save_uploaded_file(upload, fname)

# -------------------------
# Sidebar: Pipeline controls
# -------------------------
st.sidebar.header("Pipeline Controls")

if st.sidebar.button("Run Full Pipeline 🚀"):
    for step in ["bronze", "silver", "gold", "train", "score"]:
        r = requests.post(f"{API}/run/{step}")
        if r.status_code == 200:
            st.sidebar.success(f"{step} ✅")
        else:
            st.sidebar.error(f"{step} ❌: {r.text}")

st.sidebar.info("Upload CSVs above, then run pipeline.")

# -------------------------
# Helper: load Parquet + export
# -------------------------
def load_parquet(path: str, nrows=None) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    try:
        df = pd.read_parquet(p)
        if nrows:
            return df.head(nrows)
        return df
    except Exception as e:
        st.error(f"Error reading {path}: {e}")
        return pd.DataFrame()

def export_buttons(df: pd.DataFrame, name: str):
    if df.empty:
        return
    csv = df.to_csv(index=False).encode("utf-8")
    excel = BytesIO()
    with pd.ExcelWriter(excel, engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name="Sheet1", index=False)
    st.download_button("⬇️ Download CSV", csv, file_name=f"{name}.csv", mime="text/csv")
    st.download_button("⬇️ Download Excel", excel.getvalue(), file_name=f"{name}.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# -------------------------
# Tabs
# -------------------------
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🏭 Machines", "🔩 Parts", "💰 Predictions"])

# ---- Overview ----
with tab1:
    st.subheader("System Overview")
    st.markdown("""
    This dashboard demonstrates a **PySpark ETL + ML pipeline** for manufacturing data.  
    - **Upload raw CSVs** into the sidebar.  
    - Run **pipeline** (Bronze → Silver → Gold → Train → Score).  
    - Explore results in tabs below.  
    """)
    st.info("Use the other tabs to view KPIs, part features, and predictions.")

# ---- Machines ----
with tab2:
    st.subheader("Machine KPIs")
    df_m = load_parquet("gold/kpi_machine/")
    if not df_m.empty:
        scrap_range = st.slider("Filter by scrap rate", 0.0, 1.0, (0.0, 1.0))
        df_m = df_m[(df_m["scrap_rate"] >= scrap_range[0]) & (df_m["scrap_rate"] <= scrap_range[1])]
        st.dataframe(df_m)

        export_buttons(df_m, "machine_kpis")

        col1, col2 = st.columns(2)
        with col1:
            st.bar_chart(df_m.set_index("machine_id")["scrap_rate"])
        with col2:
            st.bar_chart(df_m.set_index("machine_id")["avg_cycle_time_sec"])
    else:
        st.info("No machine KPI data yet. Run pipeline first.")

# ---- Parts ----
with tab3:
    st.subheader("Per-Part Features")
    df_p = load_parquet("gold/features_per_part/")
    if not df_p.empty:
        mats = st.multiselect("Filter by material", df_p["material"].dropna().unique())
        if mats:
            df_p = df_p[df_p["material"].isin(mats)]
        st.dataframe(df_p.head(200))

        export_buttons(df_p, "part_features")

        st.scatter_chart(df_p, x="complexity_idx", y="avg_cycle_time_sec")
    else:
        st.info("No part feature data yet. Run pipeline first.")

# ---- Predictions ----
with tab4:
    st.subheader("Predicted Costs")
    df_pred = load_parquet("gold/predictions/")
    if not df_pred.empty:
        minc, maxc = float(df_pred["pred_cost"].min()), float(df_pred["pred_cost"].max())
        cost_range = st.slider("Filter by predicted cost", minc, maxc, (minc, maxc))
        df_pred = df_pred[(df_pred["pred_cost"] >= cost_range[0]) & (df_pred["pred_cost"] <= cost_range[1])]
        st.dataframe(df_pred.head(200))

        export_buttons(df_pred, "predictions")

        

        fig, ax = plt.subplots()
        df_pred["pred_cost"].hist(ax=ax, bins=30, color="purple", alpha=0.7)
        ax.set_title("Predicted Cost Distribution")
        ax.set_xlabel("Predicted Cost")
        ax.set_ylabel("Count")
        st.pyplot(fig)
    else:
        st.info("No predictions yet. Run train + score.")
