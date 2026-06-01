import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.multioutput import MultiOutputRegressor
import joblib
import os

def train_surrogate_model(data_path='data/synthetic_telemetry.csv', model_save_path='models/xgboost_surrogate.pkl'):
    print(f"Loading telemetry data from {data_path}...")
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: Could not find {data_path}. Please run 01_data_generator.py first.")
        return

    print("Preparing features and targets for the Surrogate Model...")
    df['next_moisture'] = df.groupby('season_id')['soil_moisture_mm'].shift(-1)
    df['next_nitrogen'] = df.groupby('season_id')['soil_nitrogen_kgha'].shift(-1)
    df['next_biomass'] = df.groupby('season_id')['biomass_kgha'].shift(-1)

    df = df.dropna()

    features = [
        'soil_moisture_mm', 'soil_nitrogen_kgha', 'biomass_kgha', 
        'irrigation_mm', 'nitrogen_kgha',                         
        'temperature_c', 'precipitation_mm'                       
    ]
    targets = ['next_moisture', 'next_nitrogen', 'next_biomass']

    X = df[features]
    y = df[targets]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training XGBoost Multi-Output Surrogate Model... (This might take a few seconds)")
    base_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=5)
    surrogate = MultiOutputRegressor(base_model)
    surrogate.fit(X_train, y_train)

    predictions = surrogate.predict(X_test)
    print("\nModel Evaluation (R^2 Score):")
    for i, target_name in enumerate(targets):
        r2 = r2_score(y_test.iloc[:, i], predictions[:, i])
        print(f" - {target_name}: {r2:.4f}")

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    joblib.dump(surrogate, model_save_path)
    print(f"\nSurrogate Model saved successfully to {model_save_path}!")

if __name__ == "__main__":
    train_surrogate_model()