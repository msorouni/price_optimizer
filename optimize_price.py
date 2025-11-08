
import pandas as pd
import mlflow
import numpy as np

# Load the trained model
run_id = "a2e19cf1a0e14723b8ebe975c6c374c5"  # Replace with the printed Run ID
model = mlflow.lightgbm.load_model(f"runs:/{run_id}/lightgbm_pricing")

# Define features
features = ['hour_of_day', 'price_diff', 'temp_c', 'volume_lag_1h', 'volume_lag_24h', 
            'avg_volume_7d', 'pct_change_price_7d', 'pct_change_volume_7d', 
            'traffic_score', 'promo_interaction']

def optimize_price(station_id, product_id, base_features, cost, model):
    # Generate candidate prices (e.g., ±5% of current price)
    candidates = np.arange(base_features['price'] * 0.95, base_features['price'] * 1.05, 0.005)
    best = {'score': -np.inf, 'price': None, 'pred_vol': None}
    
    for p in candidates:
        feats = base_features.copy()
        feats['price_diff'] = p - feats['competitor_price']
        feats['promo_interaction'] = feats['promo_flag'] * feats['price_diff']
        pred_vol = model.predict(feats[features].values.reshape(1, -1))[0]
        score = (p - cost) * pred_vol  # Revenue = (price - cost) * volume
        if 0 <= p - feats['competitor_price'] <= 0.05 and score > best['score']:  # Constraint
            best = {'price': p, 'pred_vol': pred_vol, 'score': score}
    
    return best

# Example usage
df = pd.read_csv("pricing_features.csv")
base_features = df[(df['station_id'] == 123) & (df['product_id'] == 1)].iloc[-1]
cost = base_features['cost']
result = optimize_price(123, 1, base_features, cost, model)
print(f"Optimal price: ${result['price']:.4f}, Predicted volume: {result['pred_vol']:.2f}, Revenue: ${result['score']:.2f}")
