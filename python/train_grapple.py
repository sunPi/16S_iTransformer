#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
16S RNA - GraPPLE Training Wrapper
"""

import pandas as pd
import pickle
import argparse
import os
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
from GraPPLE.models import GraPPLE  # after installing the repo
from utils import load_cfg

parser = argparse.ArgumentParser(description="16S RNA - Train with GraPPLE")
parser.add_argument('-t', '--trdata', type=str, required=True)
parser.add_argument('-l', '--label', type=str, required=True)
parser.add_argument('--epochs', type=int, default=30)
args = parser.parse_args()

data_file = args.trdata
label = args.label


data_File = ""
label


if data_file.endswith(".pkl"):
    df = pickle.load(open(data_file, "rb"))
else:
    df = pd.read_csv(data_file)

feature_cols = [c for c in df.columns if c.startswith("X")]
X = df[feature_cols].to_numpy(dtype="float32")
y = df["Y"].to_numpy()

le = LabelEncoder()
y_enc = le.fit_transform(y)
num_labels = len(le.classes_)

# PyTorch dataset
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y_enc, dtype=torch.long)

dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32)

# Init GraPPLE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GraPPLE(input_dim=X.shape[1], num_classes=num_labels).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(args.epochs):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

    # Eval
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = torch.argmax(model(xb), dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(yb.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    print(f"Epoch {epoch+1}/{args.epochs} - Test Acc: {acc:.4f}")

# Save
script_dir = os.path.dirname(os.path.abspath(__file__))
config = load_cfg(os.path.dirname(script_dir) + "/config.cfg")
ROOT_DIR = config["ROOT_DIR"]
dname = data_file.split('/')[8]
model_dir = Path(ROOT_DIR, 'results', 'models', label, 'grapple', dname)

os.makedirs(model_dir, exist_ok=True)
torch.save(model.state_dict(), Path(model_dir, "model.pth"))
with open(Path(model_dir, "label_encoder.pkl"), "wb") as f:
    pickle.dump(le, f)

print("✅ GraPPLE model saved to " + model_dir.as_posix())

