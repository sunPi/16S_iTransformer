#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
16S RNA - XGBoost Training
"""

import pandas as pd
import numpy as np
import pickle
import argparse
import os
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from xgboost import XGBClassifier
from utils import load_cfg

# ==============
# 1. Parse args
# ==============
parser = argparse.ArgumentParser(description="16S RNA - Train with XGBoost")
parser.add_argument('-t', '--trdata', type=str, required=True, help='Training data file (.csv or .pkl)')
parser.add_argument('-l', '--label', type=str, required=True, help='single or multi')
parser.add_argument('--epochs', type=int, default=100, help='Number of boosting rounds')
parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
parser.add_argument('--depth', type=int, default=6, help='Max tree depth')
args = parser.parse_args()

data_file = args.trdata
label = args.label

# data_file = "/home/jr453/Documents/Projects/Reem_16s_RNA_classification/16S_iTransformer/data/16S_RNA/singlelabel/silva_kingdom/train/train_silva_kingdom.pkl"
# config    = load_cfg("/home/jr453/Documents/Projects/Reem_16s_RNA_classification/16S_iTransformer/config.cfg")
# label = "single"
# epochs = 100
# lr = 0.1
# depth = 6

label  = args.label
epochs = args.epochs
lr     = args.lr
depth  = args.depth

# ==============
# 2. Load data
# ==============
if data_file.endswith(".pkl"):
    df = pickle.load(open(data_file, "rb"))
else:
    df = pd.read_csv(data_file)

feature_cols = [c for c in df.columns if c.startswith("X")]
X = df[feature_cols].to_numpy(dtype=np.float32)
y = df["Y"].to_numpy()

le = LabelEncoder()
y_enc = le.fit_transform(y)
num_labels = len(le.classes_)

# Train/test split
n = len(X)
split = int(0.8 * n)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y_enc[:split], y_enc[split:]

le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc = le.transform(y_test)

# y_train_corrected = y_train - y_train.min()
# y_test_corrected = y_test - y_train.min()

# num_labels = y_train_corrected.max() + 1

# ==============
# 3. XGBoost
# ==============
model = XGBClassifier(
    objective="multi:softmax",
    num_class=num_labels,
    n_estimators=epochs,
    learning_rate=lr,
    max_depth=depth,
    tree_method="hist",
    use_label_encoder=False,
    eval_metric="mlogloss",
    verbosity=1,
)

num_labels = len(le.classes_)

model.fit(X_train, y_train_enc, verbose=True)

y_pred_enc = model.predict(X_test)
y_pred = le.inverse_transform(y_pred_enc)  # back to original labels

# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)

# Metrics
acc = accuracy_score(y_test, y_pred)
prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted")

print(f"✅ XGBoost results: Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}")

# ==============
# 4. Save
# ==============
script_dir = os.path.dirname(os.path.abspath(__file__))
config = load_cfg(os.path.dirname(script_dir) + "/config.cfg")
ROOT_DIR = config["ROOT_DIR"]

split_path = data_file.split('/')
dname      = [x for x in split_path if x.startswith('silva_')][0]
model_dir = Path(ROOT_DIR, 'results', 'models', label, 'xgboost', dname)

os.makedirs(model_dir, exist_ok=True)

with open(Path(model_dir, "xgboost_model.pkl"), "wb") as f:
    pickle.dump(model, f)
with open(Path(model_dir, "label_encoder.pkl"), "wb") as f:
    pickle.dump(le, f)

print("✅ XGBoost model saved to " + model_dir.as_posix())

