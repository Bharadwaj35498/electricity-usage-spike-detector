import streamlit as st
import pandas as pd

from src.real_data_loader import load_real_data
from src.feature_engineering import add_rolling_features
from src.zscore_detector import detect_spikes
from src.isolation_forest import detect_anomalies

# --------------------
# Page config
# --------------------
st.set_page_config(
    page_title="Electricity Usage Spike Detector",
    layout="wide"
)

st.title("Electricity Usage Spike Detection Dashboard")

st.markdown("""
This dashboard detects **abnormal spikes in electricity usage** using:
- Statistical methods (Z-score)
- Machine Learning (Isolation Forest)
""")

# --------------------
# Sidebar controls
# --------------------
st.sidebar.header("Settings")

z_threshold = st.sidebar.slider(
    "Z-Score Threshold",
    min_value=2.0,
    max_value=4.0,
    value=3.0,
    step=0.1
)

show_zscore = st.sidebar.checkbox("Show Z-Score Spikes", value=True)
show_iforest = st.sidebar.checkbox("Show Isolation Forest Anomalies", value=True)

# --------------------
# Load & process data
# --------------------
df = load_real_data()
df = add_rolling_features(df)

# Drop rows created by rolling window
df = df.dropna().reset_index(drop=True)


# Apply Z-score logic
df["z_score"] = (df["usage"] - df["rolling_mean"]) / df["rolling_std"]
df["zscore_spike"] = df["z_score"].abs() > z_threshold
df["zscore_spike"] = df["zscore_spike"].fillna(False)

# Apply Isolation Forest
df = detect_anomalies(df)


# --------------------
# Plot
# --------------------
st.subheader("Electricity Usage Over Time")

st.line_chart(
    df.set_index("date")[["usage"]]
)


# Overlay anomalies
if show_zscore:
    st.subheader("Z-Score Detected Spikes")
    st.dataframe(
        df[df["zscore_spike"]][["date", "usage", "z_score"]]
        .sort_values("usage", ascending=False)
    )

if show_iforest:
    st.subheader("Isolation Forest Anomalies")
    st.dataframe(
        df[df["if_anomaly"] == 1][["date", "usage"]]
        .sort_values("usage", ascending=False)
    )

# --------------------
# Summary metrics
# --------------------
st.subheader("Summary")

col1, col2, col3 = st.columns(3)

col1.metric("Total Days", len(df))
col2.metric("Z-Score Spikes", df["zscore_spike"].sum())
col3.metric("Isolation Forest Anomalies", df["if_anomaly"].sum())
