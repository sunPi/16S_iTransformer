#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate trained 16S Transformer model
Computes Accuracy, Precision, Recall, F1
Plots confusion matrix
Created on Wed Sep 10 20:56:48 2025

@author: jr453
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from pathlib import Path

# ========================
# 1. Load model + encoder
# ========================
class SimpleTransformerClassifier(nn.Module):
    def __init__(self, input_dim, num_labels, d_model=128, nhead=8, num_layers=2, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, num_labels)

    def forward(self, x):
        x = self.embedding(x).unsqueeze(0)  # [1, batch, d_model]
        x = self.transformer_encoder(x)
        x = x.squeeze(0)
        return self.classifier(x)

# Load label encoder
ROOT_DIR = Path("/home/jr453/Documents/Projects/Reem_16s_RNA_classification/")
label          = "silva_species"
model_dir = Path(ROOT_DIR, 'results', 'models', 'single', '16s_transformer', label)
file             = "test_" + label + ".pkl"


with open(Path(model_dir, 'label_encoder.pkl'), "rb") as f:
    le: LabelEncoder = pickle.load(f)

num_labels = len(le.classes_)

# Load evaluation dataset (50 cut sequences, pickled/CSV format)
eval_file = Path(ROOT_DIR, 'data', 'singlelabel', label , 'test', file)  # replace with your held-out 50 seqs

if eval_file.as_posix().endswith(".pkl"):
    df_eval = pickle.load(open(eval_file, "rb"))
else:
    df_eval = pd.read_csv(eval_file)

feature_cols = [c for c in df_eval.columns if c.startswith("X")]
X_eval = df_eval[feature_cols].to_numpy(dtype=np.float32)
y_eval = le.transform(df_eval["Y"])  # encode with same label encoder

X_eval = torch.tensor(X_eval, dtype=torch.float32)
y_eval = torch.tensor(y_eval, dtype=torch.long)

# Load trained model
input_dim = X_eval.shape[1]
model = SimpleTransformerClassifier(input_dim=input_dim, num_labels=num_labels)
model.load_state_dict(torch.load(Path(model_dir, 'model.pth'), map_location="cpu"))
model.eval()

# ========================
# 2. Inference
# ========================
with torch.no_grad():
    logits = model(X_eval)
    preds = torch.argmax(F.softmax(logits, dim=1), dim=1).numpy()

y_true = y_eval.numpy()
y_pred = preds

# ========================
# 3. Metrics
# ========================
acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

print("Evaluation Results:")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1-score:  {f1:.4f}")

# ========================
# 4. Confusion Matrix
# ========================
# cm = confusion_matrix(y_true, y_pred)
# plt.figure(figsize=(10, 8))
# sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
#             xticklabels=le.classes_,
#             yticklabels=le.classes_)
# plt.xlabel("Predicted")
# plt.ylabel("True")
# plt.title("Confusion Matrix - 16S Transformer Evaluation")
# plt.tight_layout()
# plt.savefig("16s_transformer/confusion_matrix.png", dpi=300)
# plt.show()

# ========================
# 6. Barplot of Metrics
# ========================
metrics = {
    "Accuracy": acc,
    "Precision": prec,
    "Recall": rec,
    "F1-score": f1
}
plt.figure(figsize=(8, 6))
sns.barplot(x=list(metrics.keys()), y=list(metrics.values()), palette="viridis")
plt.ylim(0, 1.05)
plt.ylabel("Score")
plt.title("Evaluation Metrics - 16S Transformer on " + label)
plt.tight_layout()
plt.savefig(Path(model_dir, 'metrics_barplot.png'), dpi=300)
plt.show()
