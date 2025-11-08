import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Parameters
n_rows = 10000
stations = [123, 456]
products = [1, 2]
start_date = datetime(2025, 6, 1)
hours_increment = timedelta(hours=1)

# Initialize data
np.random.seed(42)
data = {
    'station_id': np.random.choice(stations, n_rows),
    'product_id': np.random.choice(products, n_rows),
    'ts': [start_date + i * hours_increment for i in range(n_rows)],
    'price': np.random.uniform(1.8, 2.0, n_rows),
    'competitor_price': 0.0,
    'cost': np.random.uniform(1.4, 1.6, n_rows),
    'temp_c': np.random.uniform(15, 30, n_rows),
    'promo_flag': np.random.choice([0, 1], n_rows, p=[0.8, 0.2]),
    'hour_of_day': 0,
    'day_of_week': 0,
    'is_holiday': 0,
    'is_weekend': 0,
    'volume_lag_1h': 0.0,
    'volume_lag_24h': 0.0,
    'avg_volume_7d': 0.0,
    'price_diff': 0.0,
    'pct_change_price_7d': 0.0,
    'pct_change_volume_7d': 0.0,
    'traffic_score': np.random.uniform(0.7, 1.0, n_rows),
    'promo_interaction': 0.0,
    'volume_next_hour': 0.0
}

# Create DataFrame
df = pd.DataFrame(data)

# Derive competitor price
df['competitor_price'] = df['price'] + np.random.uniform(-0.05, 0.05, n_rows)

# Derive volume with stronger correlations
df['volume'] = (
    150 +
    (df['temp_c'] - 15) * 15 -  # Stronger temp effect
    (df['price'] - df['competitor_price']) * 1000 +  # Much stronger price_diff effect
    df['promo_flag'] * 50 +  # Stronger promo effect
    df['hour_of_day'] * 2 +  # Hour effect (e.g., peak hours)
    np.random.normal(0, 3, n_rows)  # Minimal noise
)
df['volume'] = df['volume'].clip(100, 250)

# Derive temporal features
df['hour_of_day'] = df['ts'].dt.hour
df['day_of_week'] = df['ts'].dt.dayofweek
df['is_holiday'] = df['ts'].dt.date.isin([datetime(2025, 7, 4).date()]).astype(int)
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
df.loc[df['is_weekend'] == 1, 'volume'] += 20

# Sort by station, product, and timestamp
df = df.sort_values(['station_id', 'product_id', 'ts'])

# Calculate lagged and derived features
for group in df.groupby(['station_id', 'product_id']):
    station_id, product_id = group[0]
    group_df = group[1]
    group_df['volume_lag_1h'] = group_df['volume'].shift(1)
    group_df['volume_lag_24h'] = group_df['volume'].shift(24)
    group_df['avg_volume_7d'] = group_df['volume'].rolling(window=168, min_periods=1).mean()
    group_df['price_diff'] = group_df['price'] - group_df['competitor_price']
    group_df['pct_change_price_7d'] = (group_df['price'] - group_df['price'].shift(168)) / group_df['price'].shift(168) * 100
    group_df['pct_change_volume_7d'] = (group_df['volume'] - group_df['volume'].shift(168)) / group_df['volume'].shift(168) * 100
    group_df['promo_interaction'] = group_df['promo_flag'] * group_df['price_diff']
    group_df['volume_next_hour'] = group_df['volume'].shift(-1)
    df.loc[group_df.index, ['volume_lag_1h', 'volume_lag_24h', 'avg_volume_7d', 'price_diff', 
                            'pct_change_price_7d', 'pct_change_volume_7d', 'promo_interaction', 
                            'volume_next_hour']] = group_df[['volume_lag_1h', 'volume_lag_24h', 
                                                            'avg_volume_7d', 'price_diff', 
                                                            'pct_change_price_7d', 'pct_change_volume_7d', 
                                                            'promo_interaction', 'volume_next_hour']]

# Fill NaN values
df = df.fillna({'volume_lag_1h': df['volume'], 'volume_lag_24h': df['volume'], 
                'avg_volume_7d': df['volume'], 'pct_change_price_7d': 0, 
                'pct_change_volume_7d': 0, 'volume_next_hour': df['volume']})

# Save to CSV
df.to_csv('pricing_features.csv', index=False)
print(f"Generated pricing_features.csv with {len(df)} rows")