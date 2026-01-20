from src.real_data_loader import load_real_data
from src.data_loader import load_data
from src.feature_engineering import add_rolling_features
from src.zscore_detector import detect_spikes
from src.isolation_forest import detect_anomalies
from src.visualize import plot_results
from src.config import OUTPUT_PATH

def main():
    
    df = load_real_data()
    df = load_data()
    df = add_rolling_features(df)
    df = detect_spikes(df)
    df = detect_anomalies(df)

    df.to_csv(OUTPUT_PATH, index=False)
    print("✅ Final results saved.")

    plot_results(df)

if __name__ == "__main__":
    main()
