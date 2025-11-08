# Pricing Optimizer

This project provides a machine learning pipeline to optimize pricing using LightGBM models tracked with MLflow.

---

## Github repo
git init
git add README.md
git commit -m "first commit"
git branch -M main
git remote add origin git@github.com:msorouni/price_optimizer.git
git push -u origin main

## 🛠 Prerequisites

- [Miniconda or Anaconda](https://docs.conda.io/en/latest/miniconda.html)
- Python 3.11 (handled via Conda environment)
- Internet access for package installation
- (Optional) Azure credentials if connecting to Azure Storage or AML

---

## ⚡ Setup Instructions

1. **Clone or navigate to the project folder**:

```bash
cd /path/to/pricing_optimizer

#Create the Conda environment:
conda env create -f environment.yml

#Activate the environment:
conda activate pricing_optimizer

#Verify installation:
python -c "import lightgbm, mlflow, pandas, numpy; print('✅ Environment ready!')"



#if not using environment.yml
conda create -n pricing_optimizer python=3.11
conda activate pricing_optimizer
conda install -c conda-forge lightgbm mlflow pandas numpy scikit-learn

python /Users/michael/Documents/python/pricing_optimizer/optimize_price_all.py

