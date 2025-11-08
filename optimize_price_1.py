import pandas as pd
import mlflow
import numpy as np

# Load the trained model (use your latest run ID)
run_id = "6e99dd58db43416388c5bc458f73bc3d"  # Replace with new if rerun
model = mlflow.lightgbm.load_model(f"runs:/{run_id}/lightgbm_pricing")

# Define features
features = ['hour_of_day', 'price_diff', 'temp_c', 'volume_lag_1h', 'volume_lag_24h', 
            'avg_volume_7d', 'pct_change_price_7d', 'pct_change_volume_7d', 
            'traffic_score', 'promo_interaction']

def optimize_price(station_id, product_id, base_features, cost, model):
    # Candidate prices: ±5% of current price in 0.5c steps
    candidates = np.arange(base_features['price'] * 0.95, base_features['price'] * 1.05, 0.005)
    best = {'score': -np.inf, 'price': None, 'pred_vol': None}
    
    for p in candidates:
        feats = base_features.copy()
        feats['price_diff'] = p - feats['competitor_price']
        feats['promo_interaction'] = feats['promo_flag'] * feats['price_diff']
        pred_vol = model.predict(feats[features].values.reshape(1, -1))[0]
        score = (p - cost) * pred_vol  # Profit objective
        if 0 <= p - feats['competitor_price'] <= 0.05 and score > best['score']:  # Constraints
            best = {'price': p, 'pred_vol': pred_vol, 'score': score}
    
    return best

# Example: Optimize for one station/product
df = pd.read_csv("pricing_features.csv")
base_features = df[(df['station_id'] == 123) & (df['product_id'] == 1)].iloc[-1]
cost = base_features['cost']
result = optimize_price(123, 1, base_features, cost, model)
print(f"Optimal price: ${result['price']:.4f}, Predicted volume: {result['pred_vol']:.2f}, Profit: ${result['score']:.2f}")

# Batch optimization for all stations/products (save to CSV)
results = []
for (station_id, product_id), group in df.groupby(['station_id', 'product_id']):
    base_features = group.iloc[-1]
    result = optimize_price(station_id, product_id, base_features, base_features['cost'], model)
    results.append({'station_id': station_id, 'product_id': product_id, 'optimal_price': result['price'], 
                    'predicted_volume': result['pred_vol'], 'expected_profit': result['score']})

pd.DataFrame(results).to_csv('optimized_prices.csv', index=False)
print("Batch optimization results saved to optimized_prices.csv")
