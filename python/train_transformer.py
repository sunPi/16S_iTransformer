# https://huggingface.co/learn/llm-course/en/chapter1/4
# https://www.datacamp.com/tutorial/building-a-transformer-with-py-torch

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  8 20:47:23 2025

@author: jr453
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

import pickle
import os
import argparse
from pathlib import Path

from utils import *

# print("Torch version:", torch.__version__)
# print("CUDA available:", torch.cuda.is_available())
# print("CUDA version:", torch.version.cuda)
# print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")

# Set up argument parser
parser = argparse.ArgumentParser(description="16S RNA Transformer - Train")
parser.add_argument('-t', '--trdata', type=str, required=True, help='File path for the training data in either .csv or .pkl.')
parser.add_argument('-l', '--label', type=str, required=True, help='Specify if single or multi-label df.')

# Parse arguments
args = parser.parse_args()

# ========================
# 1. Load data (Pickle or CSV)
# ========================
# data_file = "/home/jr453/Documents/Projects/Reem_16s_RNA_classification/Reem_Taxonomy_Challenge/data/16S_RNA/singlelabel/silva_species/train/train_silva_species.pkl"
# label       = "singlelabel"
# config    = load_cfg("/home/jr453/Documents/Projects/Reem_16s_RNA_classification/Reem_Taxonomy_Challenge/config.cfg")

data_file = args.trdata
label       = args.label

if data_file.endswith(".pkl"):
    df = pickle.load(open(data_file, "rb"))
else:
    df = pd.read_csv(data_file)

# Features: all columns starting with X
feature_cols = [c for c in df.columns if c.startswith("X")]
X = df[feature_cols].to_numpy(dtype=np.float32)
y = df["Y"].to_numpy()

# Encode labels
le = LabelEncoder()
y_enc = le.fit_transform(y)
num_labels = len(le.classes_)

# ========================
# 2. Dataset class
# ========================
class OneHotDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

dataset = OneHotDataset(X, y_enc)

# Train/test split (80/20)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

# ========================
# 3. Transformer model
# ========================
class SimpleTransformerClassifier(nn.Module):
    def __init__(self, input_dim, num_labels, d_model=128, nhead=8, num_layers=2, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
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
        # x shape: [batch, seq_len*5]
        # Treat the entire flattened sequence as "sequence length 1" for simplicity
        x = self.embedding(x).unsqueeze(0)  # [seq_len=1, batch, d_model]
        x = self.transformer_encoder(x)
        x = x.squeeze(0)
        logits = self.classifier(x)
        return logits

# Instantiate model
input_dim = X.shape[1]  # number of one-hot columns
model = SimpleTransformerClassifier(input_dim=input_dim, num_labels=num_labels)

# ========================
# 4. Training
# ========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()
num_epochs = 50

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * xb.size(0)
    epoch_loss = running_loss / len(train_loader.dataset)
    
    # Evaluate on test set
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            preds = torch.argmax(F.softmax(logits, dim=1), dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(yb.cpu().numpy())
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    acc = accuracy_score(all_labels, all_preds)
    
    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f} - Test Acc: {acc:.4f}")

# ========================
# 5. Save model + label encoder
# ========================
script_dir = os.path.dirname(os.path.abspath(__file__))
config = load_cfg(os.path.dirname(script_dir) + "/config.cfg")

ROOT_DIR   = config["ROOT_DIR"]
split_path = data_file.split('/')
dname      = [x for x in split_path if x.startswith('silva_')][0]

model_dir = Path(ROOT_DIR, 'results', 'models', label, '16s_transformer', dname)

os.makedirs(model_dir, exist_ok=True)

torch.save(model.state_dict(), Path(model_dir, "model.pth"))
with open(Path(model_dir, "label_encoder.pkl"), "wb") as f:
    pickle.dump(le, f)

print("✅ Training finished, model saved in " + model_dir.as_posix())
