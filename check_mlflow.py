import mlflow
import os

mlruns_path = "./mlruns"
for exp in os.listdir(mlruns_path):
    exp_path = os.path.join(mlruns_path, exp)
    if os.path.isdir(exp_path):
        print("Experiment:", exp)
        for run in os.listdir(exp_path):
            run_path = os.path.join(exp_path, run, "artifacts")
            if os.path.exists(run_path):
                print("  Run ID:", run)
                print("    Artifacts:", os.listdir(run_path))
