from sklearn.ensemble import IsolationForest
from src.config import ISOLATION_CONTAMINATION, RANDOM_STATE

def detect_anomalies(df):
    model = IsolationForest(
        contamination=ISOLATION_CONTAMINATION,
        random_state=RANDOM_STATE
    )

    df["if_anomaly"] = model.fit_predict(df[["usage"]])
    df["if_anomaly"] = df["if_anomaly"].map({1: 0, -1: 1})
    return df
