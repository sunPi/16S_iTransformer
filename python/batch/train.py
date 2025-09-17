#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
16S RNA Transformer - Train on PKL Batches
Author: jr453
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
import os
from pathlib import Path
import argparse

# ========================
# Argparse
# ========================
parser = argparse.ArgumentParser(description="16S RNA Transformer - Train")
parser.add_argument(
    '-b', '--batches', type=str, nargs='+', required=True,
    help='Sorted list of .pkl batch files for training (space-separated).'
)
parser.add_argument(
    '-l', '--label', type=str, required=True,
    help='Specify if single or multi-label df.'
)
parser.add_argument(
    '-e', '--epochs', type=int, default=10,
    help='Number of training epochs (default: 10).'
)
parser.add_argument(
    '-bs', '--batch_size', type=int, default=32,
    help='Mini-batch size within each PKL batch (default: 32).'
)
args = parser.parse_args()

batch_files = args.batches
label = args.label
num_epochs = args.epochs
mini_batch_size = args.batch_size

# ========================
# Dataset class
# ========================
class OneHotDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ========================
# Transformer model
# ========================
class SimpleTransformerClassifier(nn.Module):
    def __init__(self, input_dim, num_labels, d_model=128, nhead=8, num_layers=2,
                 dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, activation="relu",
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, num_labels)

    def forward(self, x):
        # x shape: [batch, features]
        x = self.embedding(x).unsqueeze(0)  # [seq_len=1, batch, d_model]
        x = self.transformer_encoder(x)
        x = x.squeeze(0)  # [batch, d_model]
        return self.classifier(x)

# ========================
# Training loop
# ========================
def train_on_batches(batch_files, num_epochs=10, mini_batch_size=32, device="cpu"):
    le = None
    model, optimizer, criterion = None, None, None

    for epoch in range(num_epochs):
        print(f"\n=== Epoch {epoch+1}/{num_epochs} ===")
        for bf in batch_files:
            print(f"📂 Loading {bf}")
            df = pickle.load(open(bf, "rb"))

            # Features and labels
            feature_cols = [c for c in df.columns if c.startswith("X")]
            X = df[feature_cols].to_numpy(dtype=np.float32)
            y = df["Y"].to_numpy()

            # Encode labels (fit once)
            if le is None:
                le = LabelEncoder()
                y_enc = le.fit_transform(y)
                num_labels = len(le.classes_)
                input_dim = X.shape[1]

                model = SimpleTransformerClassifier(input_dim, num_labels)
                model.to(device)
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
                criterion = nn.CrossEntropyLoss()
            else:
                y_enc = le.transform(y)

            dataset = OneHotDataset(X, y_enc)
            loader = DataLoader(dataset, batch_size=mini_batch_size, shuffle=True)

            # Train on this batch
            model.train()
            running_loss = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * xb.size(0)

            avg_loss = running_loss / len(dataset)
            print(f"Batch {os.path.basename(bf)} - Loss: {avg_loss:.4f}")

    return model, le

# ========================
# Run training
# ========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, le = train_on_batches(batch_files, num_epochs=num_epochs,
                             mini_batch_size=mini_batch_size, device=device)

# ========================
# Save model + label encoder
# ========================
ROOT_DIR = "/home/jr453/Documents/Projects/Reem_16s_RNA_classification/"
dname = Path(batch_files[0]).parent.name  # use folder name of batches
model_dir = Path(ROOT_DIR, 'results', 'models', label, '16s_transformer', dname)

os.makedirs(model_dir, exist_ok=True)
torch.save(model.state_dict(), Path(model_dir, "model.pth"))
with open(Path(model_dir, "label_encoder.pkl"), "wb") as f:
    pickle.dump(le, f)

print("✅ Training finished, model saved in " + model_dir.as_posix())
