#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 00:29:41 2025

@author: jr453
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generic evaluation utilities for 16S classification models.
Computes Accuracy, Precision, Recall, F1, and saves a metrics barplot.
Compatible with XGBoost, Transformers, or any sklearn-style classifier.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score
)
from pathlib import Path

def compute_metrics(y_true, y_pred):
    """Compute standard evaluation metrics."""
    return {
        "Accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "Precision": round(precision_score(y_true, y_pred, average="weighted", zero_division=0), 4),
        "Recall":    round(recall_score(y_true, y_pred, average="weighted", zero_division=0), 4),
        "F1-score":  round(f1_score(y_true, y_pred, average="weighted", zero_division=0), 4)
    }

def save_metrics(metrics, output_dir):
    """Save metrics to CSV and create a barplot."""
    output_dir = Path(output_dir)
    csv_path = output_dir / "metrics.csv"
    png_path = output_dir / "metrics_barplot.png"

    # Save CSV
    df = pd.DataFrame([metrics])
    df.to_csv(csv_path, index=False)

    # Save barplot
    plt.figure(figsize=(6, 4))
    sns.barplot(x=list(metrics.keys()), y=list(metrics.values()), palette="viridis")
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.title("Evaluation Metrics")
    plt.tight_layout()
    plt.savefig(png_path, dpi=300)
    plt.close()

def evaluate_model(model, X_eval, y_eval, label_encoder=None, output_dir="."):
    """
    Compute evaluation metrics and save plots for any trained model.
    - model: fitted model with .predict()
    - X_eval: features (numpy or pandas)
    - y_eval: true labels (encoded or raw)
    - label_encoder: optional LabelEncoder to decode predictions
    """
    # Predict
    y_pred = model.predict(X_eval)

    # Decode if necessary
    if label_encoder is not None:
        y_pred = label_encoder.inverse_transform(y_pred)
        y_eval = label_encoder.inverse_transform(y_eval)

    # Metrics
    metrics = compute_metrics(y_eval, y_pred)
    print("\nEvaluation Results:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    # Save plots and CSV
    save_metrics(metrics, output_dir)
    return metrics
