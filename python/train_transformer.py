# https://huggingface.co/learn/llm-course/en/chapter1/4
# https://www.datacamp.com/tutorial/building-a-transformer-with-py-torch

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
16S RNA Transformer - Train on Batches
Author: jr453
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import pickle
import argparse
import os
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from utils import *

# ========== Dataset ==========
class OneHotDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

# ========== Model ==========
class SimpleTransformerClassifier(nn.Module):
    def __init__(self, input_dim, num_labels, d_model=128, nhead=8, num_layers=2, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation="relu")
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.classifier = nn.Linear(d_model, num_labels)
    def forward(self, x):
        x = self.embedding(x).unsqueeze(0)
        x = self.transformer_encoder(x)
        return self.classifier(x.squeeze(0))

# ========== Train Function ==========
def train_on_batches(batch_files, num_epochs=5, batch_size=32, device="cpu"):
    le, model = None, None
    optimizer, criterion = None, nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        print(f"\n=== Epoch {epoch+1}/{num_epochs} ===")
        for bf in batch_files:
            print(f"📂 Loading {bf}")
            df = pickle.load(open(bf, "rb"))
            feature_cols = [c for c in df.columns if c.startswith("X")]
            X = df[feature_cols].to_numpy(dtype=np.float32)
            y = df["Y"].to_numpy()

            if le is None:
                le = LabelEncoder()
                y_enc = le.fit_transform(y)
                num_labels = len(le.classes_)
                input_dim = X.shape[1]
                model = SimpleTransformerClassifier(input_dim, num_labels)
                model.to(device)
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            else:
                y_enc = le.transform(y)

            dataset = OneHotDataset(X, y_enc)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

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

# ========== Main ==========
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="16S RNA Transformer - Train on Batches")
    parser.add_argument('-p', '--path', type=str, required=True, help='Directory with batch .pkl files.')
    parser.add_argument('-l', '--label', type=str, required=True, help='Taxonomic level (e.g., species).')
    parser.add_argument('-e', '--epochs', type=int, default=5, help='Number of epochs.')
    args = parser.parse_args()

    batch_dir = Path(args.path)
    batch_files = sorted(batch_dir.glob(f"*_{args.label}_batch*.pkl"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, le = train_on_batches(batch_files, num_epochs=args.epochs, device=device)
    
    # Load in script dir and config file    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cfg_path   = os.path.dirname(script_dir) + "/config.cfg"
    config     = load_cfg(cfg_path)
    
    ROOT_DIR   = config["ROOT_DIR"]
    split_path = data_file.split('/')
    dname      = [x for x in split_path if x.startswith('silva_')][0]

    config["LABEL"] = args.label
    config["TAXA"]  = dname

    update_config(config_file=cfg_path, config_dict=config)

    model_dir = Path(ROOT_DIR, 'results', 'models', label, '16s_transformer', dname)

    os.makedirs(model_dir, exist_ok=True)
    
    out_dir = Path(batch_dir, "trained_models")
    out_dir.mkdir(exist_ok=True)
    torch.save(model.state_dict(), out_dir / f"{args.label}_transformer.pth")
    with open(out_dir / f"{args.label}_label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)

    print(f"✅ Training finished, model saved in {out_dir}")
