import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from mlflow import log_model, log_metric

# Load data from CSV instead of Spark
df = pd.read_csv("pricing_features.csv")

# Time-aware split (sort by timestamp to respect temporal order)
df = df.sort_values('ts')
train, val = train_test_split(df, test_size=0.2, shuffle=False)

# Define features and target
features = ['hour_of_day', 'price_diff', 'temp_c', 'volume_lag_1h', 'volume_lag_24h', 
            'avg_volume_7d', 'pct_change_price_7d', 'pct_change_volume_7d', 
            'traffic_score', 'promo_interaction']
target = 'volume_next_hour'

# Prepare LightGBM datasets
train_ds = lgb.Dataset(train[features], train[target])
val_ds = lgb.Dataset(val[features], val[target])

# Model parameters
params = {'objective': 'regression', 'metric': 'rmse', 'num_leaves': 31}

# Train model with early stopping
model = lgb.train(params, train_ds, valid_sets=[val_ds], callbacks=[lgb.early_stopping(10)])

# Log metrics and model to MLflow
log_metric('rmse', model.best_score['valid_0']['rmse'])
log_model(model, 'lightgbm_pricing')