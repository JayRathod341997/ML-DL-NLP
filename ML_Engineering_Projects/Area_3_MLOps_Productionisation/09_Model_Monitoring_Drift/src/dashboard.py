"""Streamlit monitoring dashboard.

Run with:
    uv run python scripts/run_dashboard.py
    OR directly:
    uv run streamlit run src/dashboard.py
"""

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def load_profile(profile_path: str) -> dict:
    if Path(profile_path).exists():
        with open(profile_path) as f:
            return json.load(f)
    return {}


def load_drift_results(results_dir: str) -> pd.DataFrame:
    results_dir = Path(results_dir)
    if not results_dir.exists():
        return pd.DataFrame()
    dfs = []
    for f in sorted(results_dir.glob("*.csv")):
        df = pd.read_csv(f)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def main():
    st.set_page_config(page_title="ML Monitoring Dashboard", layout="wide")
    st.title("ML Model Monitoring Dashboard")

    # Sidebar config
    with st.sidebar:
        st.header("Configuration")
        profile_path = st.text_input("Reference Profile", value="data/reference_profile.json")
        results_dir = st.text_input("Drift Results Dir", value="data/drift_results")
        psi_warning = st.slider("PSI Warning Threshold", 0.0, 0.5, 0.1, 0.05)
        psi_critical = st.slider("PSI Critical Threshold", 0.1, 1.0, 0.25, 0.05)

    # Reference Profile Summary
    profile = load_profile(profile_path)
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Reference Samples", profile.get("num_samples", "N/A"))
    with col2:
        num_cols = len(profile.get("columns", {}))
        st.metric("Monitored Features", num_cols)

    st.divider()

    # Drift Results
    st.subheader("Feature Drift Results")
    drift_df = load_drift_results(results_dir)

    if drift_df.empty:
        st.info("No drift results found. Run `scripts/monitor_batch.py` first.")
    else:
        if "psi_score" in drift_df.columns:
            fig = px.bar(
                drift_df,
                x="feature",
                y="psi_score",
                color="is_drift",
                title="PSI by Feature",
                color_discrete_map={True: "red", False: "green"},
            )
            fig.add_hline(y=psi_warning, line_dash="dash", line_color="orange", annotation_text="Warning")
            fig.add_hline(y=psi_critical, line_dash="dash", line_color="red", annotation_text="Critical")
            st.plotly_chart(fig, use_container_width=True)

        st.dataframe(drift_df)

    st.divider()

    # Alerts
    st.subheader("Recent Alerts")
    alerts_path = Path("data/alerts.jsonl")
    if alerts_path.exists():
        lines = alerts_path.read_text().strip().split("\n")
        alerts = [json.loads(l) for l in lines if l.strip()]
        alerts_df = pd.DataFrame(alerts)
        st.dataframe(alerts_df, use_container_width=True)
    else:
        st.info("No alerts recorded yet.")


if __name__ == "__main__":
    main()
