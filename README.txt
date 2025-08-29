# Local RandomForest Runner (Subiksha03 Dataset)

This package lets you run a RandomForest baseline experiment locally, using the Subiksha03 ransomware dataset from GitHub.

## Steps

```bash
# 1. unzip
unzip rf_experiment.zip
cd rf_experiment

# 2. setup virtual env
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. install deps
pip install -r requirements.txt

# 4. run experiment
python run_experiment.py --runs 5 --test-size 0.25
```

## Outputs
- dataset.csv
- metrics.csv (Accuracy, Precision, Recall, F1 for each run)
- confusion_matrix_run*.png (confusion matrices)
- report.md (Markdown summary)
