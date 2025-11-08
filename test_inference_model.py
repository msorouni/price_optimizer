import mlflow
import pandas as pd
import numpy as np

# Load the model
run_id = "2e061f727dd74be3be4c53e1a8d59beb"
model = mlflow.lightgbm.load_model(f"runs:/{run_id}/lightgbm_pricing")

# Define features
features = ['hour_of_day', 'price_diff', 'temp_c', 'volume_lag_1h', 'volume_lag_24h', 
            'avg_volume_7d', 'pct_change_price_7d', 'pct_change_volume_7d', 
            'traffic_score', 'promo_interaction']

# Load sample data
df = pd.read_csv("pricing_features.csv")
sample = df[features].iloc[0:1]  # First row as sample
print(f"{sample}")
# Predict
#print(model.predict(sample))
print(model.predict(sample))
predicted_volume = model.predict(sample)[0]
print(f"predicted_volume: {predicted_volume}")
actual_volume = df['volume_next_hour'].iloc[0]

print(f"Sample features: {sample.iloc[0].to_dict()}")
print(f"Predicted volume: {predicted_volume:.2f} liters")
print(f"Actual volume: {actual_volume:.2f} liters")
print(f"Error: {abs(predicted_volume - actual_volume):.2f} liters (RMSE context: 35.35)")