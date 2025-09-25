#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
16S RNA Transformer - Incremental Training with Optional Class Expansion
Author: jr453
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import pickle
import argparse
import os
from pathlib import Path

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
        x = self.embedding(x).unsqueeze(0)  # [seq_len=1, batch, d_model]
        x = self.transformer_encoder(x)
        x = x.squeeze(0)  # [batch, d_model]
        return self.classifier(x)


# ========================
# Training function
# ========================
def train(model, train_loader, test_loader, optimizer, criterion, device, num_epochs=10):
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

        # Evaluate
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

    return model


# ========================
# Main
# ========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="16S RNA Transformer - Incremental Training")
    parser.add_argument('-t', '--trdata', type=str, required=True, help='Training data file (.pkl or .csv)')
    parser.add_argument('-m', '--model', type=str, required=True, help='Path to existing model .pth (if resuming)')
    parser.add_argument('-l', '--labelencoder', type=str, required=True, help='Path to label encoder pickle')
    parser.add_argument('-o', '--outdir', type=str, required=True, help='Output directory for updated model')
    parser.add_argument('-e', '--expansion', action="store_true", help='Enable class expansion if new labels exist')
    parser.add_argument('-n', '--epochs', type=int, default=10, help='Number of epochs')
    args = parser.parse_args()

    # Load data
    if args.trdata.endswith(".pkl"):
        df = pickle.load(open(args.trdata, "rb"))
    else:
        df = pd.read_csv(args.trdata)

    feature_cols = [c for c in df.columns if c.startswith("X")]
    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = df["Y"].to_numpy()

    # Load old label encoder
    with open(args.labelencoder, "rb") as f:
        old_le = pickle.load(f)

    old_classes = list(old_le.classes_)

    # Fit new encoder (might include new classes)
    new_le = LabelEncoder()
    y_enc = new_le.fit_transform(y)
    new_classes = list(new_le.classes_)

    # Load model
    input_dim = X.shape[1]
    old_num_labels = len(old_classes)
    new_num_labels = len(new_classes)

    model = SimpleTransformerClassifier(input_dim, old_num_labels)
    model.load_state_dict(torch.load(args.model, map_location="cpu"))

    if args.expansion and new_num_labels > old_num_labels:
        print(f"⚡ Expanding classes from {old_num_labels} to {new_num_labels}")
        old_classifier = model.classifier

        new_classifier = nn.Linear(old_classifier.in_features, new_num_labels)
        with torch.no_grad():
            new_classifier.weight[:old_num_labels] = old_classifier.weight
            new_classifier.bias[:old_num_labels] = old_classifier.bias

        model.classifier = new_classifier

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    dataset = OneHotDataset(X, y_enc)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    model = train(model, train_loader, test_loader, optimizer, criterion, device, num_epochs=args.epochs)

    # Save updated model + encoder
    os.makedirs(args.outdir, exist_ok=True)
    torch.save(model.state_dict(), Path(args.outdir, "model.pth"))
    with open(Path(args.outdir, "label_encoder.pkl"), "wb") as f:
        pickle.dump(new_le, f)

    print(f"✅ Training complete. Model saved to {args.outdir}")
