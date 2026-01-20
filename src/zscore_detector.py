from src.config import ZSCORE_THRESHOLD

def detect_spikes(df):
    df["z_score"] = (df["usage"] - df["rolling_mean"]) / df["rolling_std"]
    df["zscore_spike"] = df["z_score"].abs() > ZSCORE_THRESHOLD
    return df
