# 🚨 Fraud Detection — LiveStream Monitor

This project implements a **credit card fraud detection system** using **CatBoostClassifier**, with an interactive **Streamlit frontend** that simulates live transactions and evaluates model performance.

## Overview

The workflow includes:

1. **Model Training (`Data.py`)**
   - Load and clean dataset (`creditcard.csv`).
   - Detect and remove duplicates and missing values.
   - Handle abnormal/negative amounts.
   - Balance classes with **classWeights**.
   - Train a **CatBoost** model with tuned hyperparameters.
   - Visualize key metrics: LogLoss, AUC, Recall.

2. **Interactive Application (`app.py` with Streamlit)**
   - Loads the trained model (`fraude_model.cbm`) and a threshold (`fraude_threshold.json`).
   - Data source options:
     - Upload a custom CSV with columns `Time`, `V1` … `V28`, `Amount`, and optionally `Class`.
     - Use the demo file (`sample_creditcard_demo.csv`).
   - Real-time controls:
     - Adjustable threshold.
     - Manual or auto-play with configurable speed.
     - **Top-K mode**: flag the most suspicious K transactions, ignoring threshold.
   - Global metrics (if `Class` column is present): confusion matrix, classification report, Precision-Recall and ROC curves.

---

## Requirements

- Python 3.9+
- Main libraries:
  ```bash
  pip install streamlit pandas numpy scikit-learn catboost imbalanced-learn matplotlib

▶ Usage
1. Train the model

Run the training script:

python Data.py


This will train and save the model (fraude_model.cbm) and plot metrics (LogLoss, AUC, Recall).

2. Run the Streamlit application

Launch the app:

streamlit run app.py


The application will open in your browser at localhost.

📊 Example Output

Transaction view: displays fraud probability per transaction compared to the threshold.

Confusion matrix: evaluates false positives and false negatives.

PR and ROC curves: visual summary of model performance.

📂 Project Structure
.
├── Data.py                    # Training script with CatBoost + SMOTE
├── app.py                     # Streamlit app (live fraud detection)
├── fraude_model.cbm           # Trained CatBoost model
├── fraude_threshold.json      # Saved threshold (JSON format)
├── sample_creditcard_demo.csv # Demo dataset
└── creditcard.csv             # Original dataset (not included due to size)

Notes

The dataset is highly imbalanced (<1% fraud cases), so class_weights and SMOTE are used to handle imbalance.

The training script (Data.py) uses GPU by default (task_type='GPU'). If no GPU is available, switch to:

model = CatBoostClassifier(task_type='CPU', ...)


The project is designed both for exploratory analysis and as a portfolio-ready demonstration.

🏆 Credits

Original dataset: Kaggle - Credit Card Fraud Detection

Model & Application: Sergio Daniel González López (serchex)