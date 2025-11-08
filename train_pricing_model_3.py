import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.lightgbm
import numpy as np
import os

# Set MLflow tracking URI to current directory
tracking_uri = f"file://{os.path.abspath('./mlruns')}"
mlflow.set_tracking_uri(tracking_uri)

# Create mlruns directory if it doesn't exist
os.makedirs("./mlruns", exist_ok=True)

# Create or get experiment
experiment_name = "pricing_optimizer"
try:
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id
    print(f"Using experiment ID: {experiment_id}")
except Exception as e:
    print(f"Error setting experiment: {e}")
    exit(1)

# Load data from CSV
try:
    df = pd.read_csv("pricing_features.csv")
except FileNotFoundError:
    print("Error: pricing_features.csv not found in current directory")
    exit(1)

# Ensure required columns exist
required_columns = ['ts', 'hour_of_day', 'price_diff', 'temp_c', 'volume_lag_1h', 
                    'volume_lag_24h', 'avg_volume_7d', 'pct_change_price_7d', 
                    'pct_change_volume_7d', 'traffic_score', 'promo_interaction', 
                    'volume_next_hour']
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    print(f"Error: Missing columns in CSV: {missing_columns}")
    exit(1)

# Time-aware split
df = df.sort_values('ts')
train, val = train_test_split(df, test_size=0.2, shuffle=False)

# Define features and target
features = ['hour_of_day', 'price_diff', 'temp_c', 'volume_lag_1h', 'volume_lag_24h', 
            'avg_volume_7d', 'pct_change_price_7d', 'pct_change_volume_7d', 
            'traffic_score', 'promo_interaction']
target = 'volume_next_hour'

# Prepare input example for MLflow
input_example = train[features].iloc[:5]
input_example[['hour_of_day', 'promo_interaction']] = input_example[['hour_of_day', 'promo_interaction']].astype('float64')

# Prepare LightGBM datasets
train_ds = lgb.Dataset(train[features], train[target])
val_ds = lgb.Dataset(val[features], val[target])

# Model parameters
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'num_leaves': 64,
    'learning_rate': 0.01,  # Lowered for better convergence
    'min_data_in_leaf': 1,
    'min_data_in_bin': 1
}

# Start MLflow run with explicit experiment ID
try:
    with mlflow.start_run(experiment_id=experiment_id) as run:
        # Print run ID
        print(f"Run ID: {run.info.run_id}")
        
        # Train model with early stopping
        model = lgb.train(params, train_ds, valid_sets=[val_ds], callbacks=[lgb.early_stopping(10)])
        
        # Log RMSE metric
        rmse = model.best_score['valid_0']['rmse']
        mlflow.log_metric('rmse', rmse)
        
        # Log model with input example
        mlflow.lightgbm.log_model(model, name='lightgbm_pricing', input_example=input_example)
        
        print(f"Model trained successfully. RMSE: {rmse}")
except Exception as e:
    print(f"Error during MLflow run: {e}")
    exit(1)
