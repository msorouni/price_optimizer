import pandas as pd
import mlflow
import numpy as np

# Experiment ID
experiment_id = "511756754799839168"
artifact_name = "lightgbm_pricing"

# Find the latest run with the model artifact
client = mlflow.tracking.MlflowClient()
# Use search_runs instead of list_run_infos
runs = client.search_runs(
    experiment_ids=[experiment_id],
    order_by=["start_time DESC"],
    max_results=50  # adjust if you have many runs
)

run_id = None
for r in runs:
    artifacts = client.list_artifacts(r.info.run_id)
    if any(a.path == artifact_name for a in artifacts):
        run_id = r.info.run_id
        break

if run_id is None:
    raise ValueError(f"No run found with artifact '{artifact_name}' in experiment {experiment_id}")

print(f"Using run {run_id} with artifact '{artifact_name}'")
model = mlflow.lightgbm.load_model(f"runs:/{run_id}/{artifact_name}")

# Features to use
features = ['hour_of_day', 'price_diff', 'temp_c', 'volume_lag_1h', 'volume_lag_24h',
            'avg_volume_7d', 'pct_change_price_7d', 'pct_change_volume_7d',
            'traffic_score', 'promo_interaction']

def optimize_price(station_id, product_id, base_features, cost, model, min_margin=0.1, max_price_diff=0.05):
    current_price = base_features['price']
    competitor_price = base_features['competitor_price']
    promo_flag = base_features['promo_flag']
    
    candidates = np.arange(current_price * 0.95, current_price * 1.05 + 0.001, 0.005)
    best = {'score': -np.inf, 'price': None, 'pred_vol': None, 'revenue': None, 'price_diff': None}
    
    valid_candidates = 0
    for p in candidates:
        price_diff = p - competitor_price
        if abs(price_diff) > max_price_diff or (p - cost) < min_margin:
            continue
        
        valid_candidates += 1
        feats = base_features.copy()
        feats['price_diff'] = price_diff
        feats['promo_interaction'] = promo_flag * price_diff
        feat_array = pd.DataFrame([feats])[features]
        
        try:
            pred_vol = model.predict(feat_array)[0]
            if pred_vol <= 0:
                continue
            
            score = (p - cost) * pred_vol
            if score > best['score']:
                best = {
                    'score': score,
                    'price': float(p),
                    'pred_vol': pred_vol,
                    'revenue': score,
                    'price_diff': price_diff
                }
        except Exception as e:
            print(f"Error predicting for price {p:.4f}: {e}")
            continue
    
    if best['price'] is None:
        return {
            'station_id': station_id,
            'product_id': product_id,
            'price': current_price,
            'pred_vol': 0,
            'revenue': 0,
            'price_diff': 0,
            'uplift': 0
        }
    return best

# Load data
try:
    df = pd.read_csv("pricing_features.csv")
except FileNotFoundError:
    print("Error: pricing_features.csv not found")
    exit(1)

# Batch optimization
results = []
for (station_id, product_id), group in df.groupby(['station_id', 'product_id']):
    if len(group) == 0:
        continue
    base_features = group.iloc[-1]
    cost = base_features['cost']
    result = optimize_price(station_id, product_id, base_features, cost, model)
    
    result['station_id'] = station_id
    result['product_id'] = product_id
    result['baseline_price'] = base_features['price']
    baseline_volume = base_features['volume_next_hour']
    result['baseline_revenue'] = (base_features['price'] - cost) * baseline_volume
    result['uplift'] = result['revenue'] - result['baseline_revenue']
    
    results.append(result)

if results:
    results_df = pd.DataFrame(results)
    results_df.to_csv('price_optimizations.csv', index=False)
    print(f"Optimized prices for {len(results)} station-product pairs")
    print(results_df[['station_id', 'product_id', 'baseline_price', 'price', 'uplift']].head())
else:
    print("No valid optimizations produced")

# Example single optimization
example_row = df[(df['station_id'] == 123) & (df['product_id'] == 1)].iloc[-1]
example_result = optimize_price(123, 1, example_row, example_row['cost'], model)
example_result['baseline_price'] = example_row['price']
example_result['baseline_revenue'] = (example_row['price'] - example_row['cost']) * example_row['volume_next_hour']
example_result['uplift'] = example_result['revenue'] - example_result['baseline_revenue']
print(f"\nExample for Station 123, Product 1:")
print(f"  Baseline: ${example_result['baseline_price']:.4f} → Revenue: ${example_result['baseline_revenue']:.2f}")
print(f"  Optimal:  ${example_result['price']:.4f} → Revenue: ${example_result['revenue']:.2f}")
print(f"  Uplift:   +${example_result['uplift']:.2f} ({(example_result['uplift']/example_result['baseline_revenue']*100):.1f}%)")
