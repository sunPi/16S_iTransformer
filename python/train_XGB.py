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
from utils import *
from eval_utils import *

# ==============
# 1. Parse args
# ==============
parser = argparse.ArgumentParser(description="16S RNA - Train with XGBoost")
parser.add_argument('-t', '--trdata', type=str, required=True, help='Training data file (.csv or .pkl)')
parser.add_argument('--epochs', type=int, default=100, help='Number of boosting rounds')
parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
parser.add_argument('--depth', type=int, default=6, help='Max tree depth')
parser.add_argument('--batch', type=str, default='False', help='Max tree depth')

args = parser.parse_args()

data_file = args.trdata
epochs    = args.epochs
lr        = args.lr
depth     = args.depth
batch     = args.batch

if batch == 'True':
    batch = True
else:
    batch = False
    
# data_file = "/home/jr453/Documents/Projects/Reem_16s_RNA_classification/16S_iTransformer/data/16S_RNA/singlelabel/silva_species/train/train_silva_species.pkl"
# eval_file = "/home/jr453/Documents/Projects/Reem_16s_RNA_classification/16S_iTransformer/data/16S_RNA/singlelabel/silva_species/test/test_silva_species.pkl"

data_file = "/home/jr453/Documents/Projects/Reem_16s_RNA_classification/16S_iTransformer/data/16S_RNA/singlelabel/batches/silva_species/train"
eval_file = "/home/jr453/Documents/Projects/Reem_16s_RNA_classification/16S_iTransformer/data/16S_RNA/singlelabel/silva_species/test/test_silva_species.pkl"

script_dir = "/home/jr453/Documents/Projects/Reem_16s_RNA_classification/16S_iTransformer/python"
label = "single"
epochs = 100
lr = 0.1
depth = 6

config     = load_cfg()
ROOT_DIR   = config["ROOT_DIR"]
FNAME      = config["FNAME"]
LABEL      = config["LABEL"]

file             = "test_" + FNAME + ".pkl"
eval_file = Path(ROOT_DIR, 'data', '16S_RNA', LABEL, FNAME , 'test', file)

# ==============
# 2. Load data
# ==============
if data_file.endswith(".pkl"):
    df = pickle.load(open(data_file, "rb"))
else:
    df = pd.read_csv(data_file)
    
def prepare_xgb_input(df, le):
    feature_cols = [c for c in df.columns if c.startswith("X")]
    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = df["Y"].to_numpy()
    
    # Encode labels
    y_enc = le.fit_transform(y)
    num_labels = len(le.classes_)
    
    return X, y_enc, num_labels

le = LabelEncoder()
X, y_enc, num_labels = prepare_xgb_input(df, le)

# ==============
# 3. Train model
# ==============
if batch:
    # Initialize XGBoost classifier
    num_estimators_per_batch = 10
    model = XGBClassifier(n_estimators=num_estimators_per_batch, random_state=42)
    model.fit(X_train, y_train)

    # Train model in batches of rounds
    for i in range(num_batches):
        model.fit(X_train, y_train, xgb_model=model.get_booster())
    
        # Make predictions on train and test data
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
    
        # Calculate and print accuracy scores
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        print(f"Batch {i+1}/{num_batches} - Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}")

else:
    model = XGBClassifier(
        objective="multi:softmax",
        num_class=num_labels,
        n_estimators=epochs,
        learning_rate=lr,
        max_depth=depth,
        tree_method="hist",
        eval_metric="mlogloss",
        verbosity=1
    )
    
    if eval_file is not None:
        if data_file.endswith(".pkl"):
            df_eval = pickle.load(open(data_file, "rb"))
        else:
            df_eval = pd.read_csv(data_file)
            
        X_eval, y_eval_enc, num_labels = prepare_xgb_input(df_eval, le)
        
        # Fit with eval_set (X_val, y_val optional)
        model.fit(
            X, y_enc,
            eval_set=[(X_eval, y_eval_enc)],  # or [(X_train, y_train), (X_val, y_val)]
            verbose=True
        )
        
    else:
        model.fit(X, y_enc, verbose=True)

# ==============
# 4. Save model
# ==============
# config["LABEL"] = args.label
# config["TAXA"]  = dname
# update_config(config_file=cfg_path, config_dict=config)

model_dir = Path(ROOT_DIR, 'results', 'models', LABEL, 'xgboost', FNAME)

os.makedirs(model_dir, exist_ok=True)

with open(Path(model_dir, "xgboost_model.pkl"), "wb") as f:
    pickle.dump(model, f)
with open(Path(model_dir, "label_encoder.pkl"), "wb") as f:
    pickle.dump(le, f)

# Suppose you already have X_eval, y_eval, and trained model
metrics = evaluate_model(
    model, 
    X_eval, 
    y_eval_enc, 
    label_encoder=le, 
    output_dir=model_dir
)

print("✅ XGBoost model saved to " + model_dir.as_posix())
