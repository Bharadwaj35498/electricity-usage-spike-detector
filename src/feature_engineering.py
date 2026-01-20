from src.config import ROLLING_WINDOW

def add_rolling_features(df):
    df["rolling_mean"] = df["usage"].rolling(ROLLING_WINDOW).mean()
    df["rolling_std"] = df["usage"].rolling(ROLLING_WINDOW).std()
    return df
