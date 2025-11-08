import pandas as pd
import mlflow
import numpy as np

# Load the trained model
run_id = "6e99dd58db43416388c5bc458f73bc3d"
model = mlflow.lightgbm.load_model(f"runs:/{run_id}/lightgbm_pricing")

# Define features
features = ['hour_of_day', 'price_diff', 'temp_c', 'volume_lag_1h', 'volume_lag_24h', 
            'avg_volume_7d', 'pct_change_price_7d', 'pct_change_volume_7d', 
            'traffic_score', 'promo_interaction']

def optimize_price(station_id, product_id, base_features, cost, model, min_margin=0.1, max_price_diff=0.05):
    """
    Optimize price for a station/product using grid search.
    
    Args:
        base_features: Row from pricing_features.csv (as dict or Series)
        cost: Supply cost per liter
        model: Trained LightGBM model
        min_margin: Minimum profit margin (price - cost)
        max_price_diff: Max difference from competitor price
    
    Returns:
        dict: Optimal price, predicted volume, expected profit
    """
    current_price = base_features['price']
    competitor_price = base_features['competitor_price']
    promo_flag = base_features['promo_flag']
    
    # Generate candidate prices (±5% of current, 0.5c steps)
    candidates = np.arange(current_price * 0.95, current_price * 1.05 + 0.001, 0.005)
    best = {'score': -np.inf, 'price': None, 'pred_vol': None, 'revenue': None, 'price_diff': None}
    print(f"Debug: Initial best = {best}")

    valid_candidates = 0
    for p in candidates:
        price_diff = p - competitor_price
        if abs(price_diff) > max_price_diff or (p - cost) < min_margin:
            continue  # Skip invalid candidates
        
        # Update features for prediction
        valid_candidates += 1
        feats = base_features.copy()
        feats['price_diff'] = price_diff
        feats['promo_interaction'] = promo_flag * price_diff
        feat_array = pd.DataFrame([feats])[features]  # Ensure DataFrame for prediction
        
        try:
            pred_vol = model.predict(feat_array)[0]
            if pred_vol <= 0:
                continue  # Skip negative predictions
            
            score = (p - cost) * pred_vol  # Profit (margin * volume)
            print(f"score = {score}")
            print(f"Debug: Initial best = {best}")

            if score > best['score']:
                best = {
                    'score': score,
                    'price': float(p),
                    'pred_vol': pred_vol,
                    'revenue': score,
                    'price_diff': price_diff
                }
                print(f"Debug: Updated best = {best}")
        except Exception as e:
            print(f"Error predicting for price {p:.4f}: {e}")
            continue
    
    print(f"Debug: Final best = {best}, Valid candidates processed: {valid_candidates}")
    
    if best['price'] is None:
        print(f"Warning: No valid price found for station {station_id}, product {product_id}")
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

# Batch optimization for all stations/products
try:
    df = pd.read_csv("pricing_features.csv")
except FileNotFoundError:
    print("Error: pricing_features.csv not found")
    exit(1)

results = []
for (station_id, product_id), group in df.groupby(['station_id', 'product_id']):
    if len(group) == 0:
        print(f"Warning: Empty group for station {station_id}, product {product_id}")
        continue

    base_features = group.iloc[-1]  # Use latest row as base
    cost = base_features['cost']
    
    result = optimize_price(station_id, product_id, base_features, cost, model)
    if result['price'] is not None:
        result['station_id'] = station_id
        result['product_id'] = product_id
        result['baseline_price'] = base_features['price']
        baseline_volume = base_features['volume_next_hour']
        result['baseline_revenue'] = (base_features['price'] - cost) * baseline_volume
        result['uplift'] = result['revenue'] - result['baseline_revenue']
        results.append(result)

# Save results
if results:
    results_df = pd.DataFrame(results)
    results_df.to_csv('price_optimizations.csv', index=False)
    print(f"Optimized prices for {len(results)} station-product pairs")
    print(results_df[['station_id', 'product_id', 'baseline_price', 'price', 'uplift']].head())
else:
    print("Error: No valid optimizations produced")

# Example single optimization
example_row = df[(df['station_id'] == 123) & (df['product_id'] == 1)].iloc[-1]
example_result = optimize_price(123, 1, example_row, example_row['cost'], model)
if example_result['price'] is not None:
    example_result['baseline_price'] = example_row['price']
    example_result['baseline_revenue'] = (example_row['price'] - example_row['cost']) * example_row['volume_next_hour']
    example_result['uplift'] = example_result['revenue'] - example_result['baseline_revenue']
    print(f"\nExample for Station 123, Product 1:")
    print(f"  Baseline: ${example_result['baseline_price']:.4f} → Revenue: ${example_result['baseline_revenue']:.2f}")
    print(f"  Optimal:  ${example_result['price']:.4f} → Revenue: ${example_result['revenue']:.2f}")
    print(f"  Uplift:   +${example_result['uplift']:.2f} ({(example_result['uplift']/example_result['baseline_revenue']*100):.1f}%)")
else:
    print("Error: Example optimization failed")