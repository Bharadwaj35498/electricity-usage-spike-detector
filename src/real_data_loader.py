import pandas as pd

def load_real_data():
    df = pd.read_csv(
        "data/electricity_data.txt",
        sep=";",
        na_values="?",
        low_memory=False
    )

    # Combine Date + Time into one datetime column
    df["datetime"] = pd.to_datetime(
        df["Date"] + " " + df["Time"],
        format="%d/%m/%Y %H:%M:%S"
    )

    df = df[["datetime", "Global_active_power"]]
    df.columns = ["date", "usage"]

    # Convert to numeric
    df["usage"] = pd.to_numeric(df["usage"])

    # Drop missing values
    df = df.dropna()

    # Aggregate to DAILY usage (important)
    df = df.resample("D", on="date").mean().reset_index()

    return df
