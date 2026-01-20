import matplotlib.pyplot as plt

def plot_results(df):
    plt.figure(figsize=(14,5))

    plt.plot(df["date"], df["usage"], label="Usage")

    plt.scatter(
        df[df["zscore_spike"]]["date"],
        df[df["zscore_spike"]]["usage"],
        color="red",
        label="Z-Score Spike"
    )

    plt.scatter(
        df[df["if_anomaly"] == 1]["date"],
        df[df["if_anomaly"] == 1]["usage"],
        color="orange",
        label="Isolation Forest Anomaly"
    )

    plt.legend()
    plt.title("Electricity Usage Spike Detection")
    plt.xlabel("Date")
    plt.ylabel("Usage Units")
    plt.show()
