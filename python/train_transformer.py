#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
16S RNA Transformer Training Script (supports full + batch training)

!!!!! BATCH TRAINING NEEDS REDEVELOPMENT - THE SPLIT SCRIPT NEEDS TO SPLIT BOTH TRAINING
!!!!! AND TESTING DATA !!!!

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
from typing import List, Optional


# ----------------------------
# Utility: load config (optional)
# ----------------------------

def load_cfg(path: str) -> dict:
    cfg = {}
    p = Path(path)
    if not p.exists():
        return cfg
    with open(p, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, v = line.split("=", 1)
                cfg[k.strip()] = v.strip()
    return cfg


def update_config(config_file: str, config_dict: dict):
    """Write a dictionary to a config file in KEY=VALUE format."""
    config_file = Path(config_file)
    with open(config_file, "w") as f:
        for key, value in config_dict.items():
            f.write(f"{key}={value}\n")
    print(f"Config saved to {config_file}")


# ----------------------------
# Dataset helper
# ----------------------------
class OneHotDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        # X: (n_samples, n_features) float32
        # y: integer-encoded labels
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ----------------------------
# Model
# ----------------------------
class SimpleTransformerClassifier(nn.Module):
    def __init__(self, input_dim: int, num_labels: int, d_model=128, nhead=8,
                 num_layers=2, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, activation="relu"
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, num_labels)

    def forward(self, x):
        # x: [batch, input_dim]
        x = self.embedding(x).unsqueeze(0)   # [seq_len=1, batch, d_model]
        x = self.transformer_encoder(x)
        x = x.squeeze(0)                     # [batch, d_model]
        return self.classifier(x)


# ----------------------------
# Training loop (supports both full and batch modes)
# ----------------------------
def train_transformer(model: nn.Module,
                      optimizer,
                      criterion,
                      device,
                      le = None,
                      num_epochs: int = 50,
                      train_loader: Optional[DataLoader] = None,
                      test_loader: Optional[DataLoader] = None,
                      batch_files: Optional[List[Path]] = None,
                      batch_size: int = 32,):

    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        total_samples = 0

        print(f"\n=== Epoch {epoch+1}/{num_epochs} ===")

        # MODE 1: standard DataLoader
        if batch_files is None:
            if train_loader is None:
                raise ValueError("train_loader is None: no training data provided")
        
            # Optional: break full dataset into smaller batches
            if batch_size < len(train_loader.dataset):
                dataset = train_loader.dataset
                loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            else:
                loader = train_loader  # already smaller than batch_size
        
            for xb, yb in loader:
                print(f"Batch X shape: {xb.shape}, Batch Y shape: {yb.shape}")
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * xb.size(0)
                total_samples += xb.size(0)
        
            epoch_loss = running_loss / total_samples

        # MODE 2: iterate over saved batch files (each file is a DataFrame)
        else:
            for bf in batch_files:
                print(f"📂 Loading {bf.name}")
                with open(bf, "rb") as f:
                    df = pickle.load(f)

                feature_cols = [c for c in df.columns if c.startswith("X")]
                X = df[feature_cols].to_numpy(dtype=np.float32)
                # assume that label encoder transformation already done externally
                # y = df["Y"].to_numpy()
                if le is None:
                    raise ValueError("le is None: no label encoder object provided")
                
                y = le.transform(df["Y"].astype(str).to_numpy())
                
                # if y is not numeric, expect it already encoded earlier
                if y.dtype == object or not np.issubdtype(y.dtype, np.integer):
                    raise TypeError("Batch labels must be integer-encoded before training.\n"
                                    "Fit a LabelEncoder across batches and transform labels in the batches,"
                                    " or let this script build the encoder and re-write encoded batches.")

                dataset = OneHotDataset(X, y)
                loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

                for xb, yb in loader:
                    xb, yb = xb.to(device), yb.to(device)
                    optimizer.zero_grad()
                    logits = model(xb)
                    loss = criterion(logits, yb)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item() * xb.size(0)
                    total_samples += xb.size(0)

            if total_samples == 0:
                epoch_loss = float('nan')
            else:
                epoch_loss = running_loss / total_samples

        # optional evaluation if test_loader provided
        if test_loader is not None:
            model.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for xb, yb in test_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    logits = model(xb)
                    preds = torch.argmax(F.softmax(logits, dim=1), dim=1)
                    all_preds.append(preds.cpu().numpy())
                    all_labels.append(yb.cpu().numpy())

            if all_preds:
                all_preds = np.concatenate(all_preds)
                all_labels = np.concatenate(all_labels)
                acc = accuracy_score(all_labels, all_preds)
                print(f"Epoch {epoch+1} - Loss: {epoch_loss:.4f} - Test Acc: {acc:.4f}")
            else:
                print(f"Epoch {epoch+1} - Loss: {epoch_loss:.4f} - Test: empty")
        else:
            print(f"Epoch {epoch+1} - Loss: {epoch_loss:.4f}")

    return model


# ----------------------------
# CLI: parse args and run
# ----------------------------
def parse_batch_files(arg: Optional[List[str]]) -> Optional[List[Path]]:
    """Handle -b/--batch_files argument. Accepts:
      - a single directory path: returns all .pkl in dir
      - one or more explicit file paths
      - glob patterns (shell usually expands these, but we support strings)
    """
    
    if arg is None:
        return None

    # if user provided a single entry that is a directory, list files
    if len(arg) == 1:
        p = Path(arg[0])
        if p.is_dir():
            files = sorted(p.glob("*.pkl"))
            return files

    # otherwise interpret each entry as a path (or pattern)
    out = []
    for entry in arg:
        # expand any glob-like patterns
        matches = list(Path().glob(entry)) if any(ch in entry for ch in "*?[]") else [Path(entry)]
        for m in matches:
            if m.exists() and m.suffix == ".pkl":
                out.append(m)
    return sorted(out)


def main():
    parser = argparse.ArgumentParser(description="16S RNA Transformer - Train")
    parser.add_argument('-t', '--trdata', type=str, required=False,
                        help='File path for the training data (.csv or .pkl).')
    parser.add_argument('-l', '--label', type=str, required=True,
                        help='Specify if single or multi-label df.')
    parser.add_argument('-b', '--batch_files', nargs='+', default=None,
                        help='Directory or list of .pkl batch files for training.')
    parser.add_argument('--batch_size', type=int, default=32, help='mini-batch size used when iterating inside each file')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--val', type=str, default=None, help='Optional validation file (.pkl or .csv)')

    args = parser.parse_args()
    
    trdata      = args.trdata
    label       = args.label
    batch_files = args.batch_files
    batch_size  = args.batch_size
    epochs      = args.epochs
    lr          = args.lr
    val         = args.val
    
    # trdata = "/home/jr453/Documents/Projects/Reem_16s_RNA_classification/16S_iTransformer/data/16S_RNA/singlelabel/silva_species/train/train_silva_species.pkl"
    # label  = 'singlelabel'
    # batch_files = '/home/jr453/Documents/Projects/Reem_16s_RNA_classification/16S_iTransformer/data/16S_RNA/singlelabel/silva_species/train/'
    # batch_size=50
    # epochs=5
    # lr=1e-3
    # val = "/home/jr453/Documents/Projects/Reem_16s_RNA_classification/16S_iTransformer/data/16S_RNA/singlelabel/silva_species/test/test_silva_species.pkl"        
    
    # if using batch mode — fit encoder on union of labels across batches
    le = None
    num_labels = None
    test_loader = None
    train_loader = None

    if batch_files is None:
        if args.trdata is None:
            raise ValueError("Either --trdata (full file) or --batch_files (folder/files) must be provided")

        #trdata = args.trdata
        df = pickle.load(open(trdata, "rb")) if trdata.endswith('.pkl') else pd.read_csv(trdata)

        feature_cols = [c for c in df.columns if c.startswith('X')]
        X = df[feature_cols].to_numpy(dtype=np.float32)
        y = df['Y'].astype(str).to_numpy()

        le = LabelEncoder()
        y_enc = le.fit_transform(y)
        num_labels = len(le.classes_)

        dataset = OneHotDataset(X, y_enc)
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

    else:
        print("Processing batches...")
        # batch mode: collect labels across all batches to fit LabelEncoder
        p = Path(batch_files)
        if p.is_dir():
            batch_files = sorted(p.glob("*.pkl"))
        
        all_labels = []
        for bf in batch_files:
            with open(bf, "rb") as f:
                df_b = pickle.load(f)
            all_labels.extend(df_b['Y'].astype(str).tolist())
            
        le = LabelEncoder()
        le.fit(np.unique(all_labels))
        num_labels = len(le.classes_)

        # If user provided a separate validation file, load and encode it
        if val is not None:
            val_file = val
            df_val = pickle.load(open(val_file, "rb")) if val_file.endswith('.pkl') else pd.read_csv(val_file)
            feature_cols = [c for c in df_val.columns if c.startswith('X')]
            X_val = df_val[feature_cols].to_numpy(dtype=np.float32)
            y_val = le.transform(df_val['Y'].astype(str).to_numpy())
            test_loader = DataLoader(OneHotDataset(X_val, y_val), batch_size=batch_size)
            
    # for bf in batch_files: # inspect sizes of batches
    #     df_b = pickle.load(open(bf, 'rb'))
    #     print(len([c for c in df_b.columns if c.startswith('X')]))
        
    # initialize model
    # derive input_dim from either df (full) or first batch
    if batch_files is None:
        input_dim = X.shape[1]
    else:
        # inspect first file
        with open(batch_files[0], "rb") as f:
            df0 = pickle.load(f)
        #input_dim = len([c for c in df0.columns if c.startswith('X')])
        feature_cols = [c for c in df0.columns if c.startswith('X')]
        input_dim = len(feature_cols)
    
    model = SimpleTransformerClassifier(input_dim=input_dim, num_labels=num_labels)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # run training
    model = train_transformer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        le=le,
        num_epochs=epochs,
        train_loader=train_loader,
        test_loader=test_loader,
        batch_files=batch_files,
        batch_size=batch_size
    )

    # save results
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cfg_path = os.path.dirname(script_dir) + "/config.cfg"
    config = load_cfg(cfg_path)
    ROOT_DIR = config.get("ROOT_DIR", os.getcwd())
    split_path = trdata.split('/')
    dname      = [x for x in split_path if x.startswith('silva_')][0]
    
    config["LABEL"] = args.label
    config["TAXA"] = dname
    update_config(cfg_path, config)

    model_dir = Path(ROOT_DIR, 'results', 'models', args.label, '16s_transformer', dname)
    os.makedirs(model_dir, exist_ok=True)

    torch.save(model.state_dict(), Path(model_dir, "model.pth"))
    with open(Path(model_dir, "label_encoder.pkl"), "wb") as f:
        pickle.dump(le, f)

    print(f"✅ Training finished, model saved in {model_dir.as_posix()}")
    print(model_dir.as_posix())


if __name__ == '__main__':
    main()
