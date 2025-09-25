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

# Encode labels
le = LabelEncoder()
y_enc = le.fit_transform(y)
num_labels = len(le.classes_)

# ==============
# 3. Train model
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

model.fit(X, y_enc, verbose=True)

# ==============
# 4. Save model
# ==============
script_dir = os.path.dirname(os.path.abspath(__file__))
cfg_path   = os.path.dirname(script_dir) + "/config.cfg"
config     = load_cfg(cfg_path)
ROOT_DIR   = config["ROOT_DIR"]

split_path = data_file.split('/')
dname      = [x for x in split_path if x.startswith('silva_')][0]


config["LABEL"] = args.label
config["TAXA"]  = dname

update_config(config_file=cfg_path, config_dict=config)

model_dir = Path(ROOT_DIR, 'results', 'models', label, 'xgboost', dname)

os.makedirs(model_dir, exist_ok=True)

with open(Path(model_dir, "xgboost_model.pkl"), "wb") as f:
    pickle.dump(model, f)
with open(Path(model_dir, "label_encoder.pkl"), "wb") as f:
    pickle.dump(le, f)

print("✅ XGBoost model saved to " + model_dir.as_posix())
