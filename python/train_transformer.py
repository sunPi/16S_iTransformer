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
from utils import *
import ast
import matplotlib.pyplot as plt

# ----------------------------
# Dataset helper
# ----------------------------
class EncodedDataset(Dataset):
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

def plot_loss_curve(train_losses, val_losses=None, save_path="loss_curve.jpg"):
    """
    Plots training (and optional validation) loss and saves as JPG.

    Args:
        train_losses (list or array): Training loss per epoch
        val_losses (list or array, optional): Validation loss per epoch
        save_path (str): Path to save the plot (JPG)
    """
    plt.figure(figsize=(8, 6))
    plt.plot(train_losses, label="Train Loss", marker='o')
    
    if val_losses is not None:
        plt.plot(val_losses, label="Val Loss", marker='o')
    
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Convergence")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)  # Save as high-res JPG
    plt.close()  # Close figure to avoid display in notebooks
    print(f"Loss curve saved to {save_path}")
    
# ----------------------------
# Training loop (supports both full and batch modes)
# ----------------------------
def train_transformer(model: nn.Module,
                      optimizer,
                      criterion,
                      le = None,
                      num_epochs: int = 50,
                      train_loader: Optional[DataLoader] = None,
                      test_loader: Optional[DataLoader] = None,
                      batch_files: Optional[List[Path]] = None,
                      batch_size: int = 32):
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    train_losses = []
    val_losses = []
    
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
            # if batch_size < len(train_loader.dataset):
            #     dataset = train_loader.dataset
            #     loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            # else:
            #     pass
            
            loader = train_loader  # already smaller than batch_size

            for xb, yb in loader:
                #print(f"Batch X shape: {xb.shape}, Batch Y shape: {yb.shape}")
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * xb.size(0)
                total_samples += xb.size(0)
        
            epoch_loss = running_loss / total_samples
            train_losses.append(epoch_loss)
            
        # MODE 2: iterate over saved batch files (each file is a DataFrame)
        else:
            for bf in batch_files:
                # print(f"📂 Loading {bf.name}")
                with open(bf, "rb") as f:
                    df = pickle.load(f)
                    # print(f"{bf.name}: {df.shape}")
                
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

                dataset = EncodedDataset(X, y)
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
                train_losses.append(epoch_loss)
            else:
                epoch_loss = running_loss / total_samples
                train_losses.append(epoch_loss)

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

    return model, train_losses

# ----------------------------
# CLI: parse args and run
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="16S RNA Transformer - Train")
    parser.add_argument('-t', '--trdata', type=str, required=False,
                        help='File path for the training data (.csv or .pkl).')
    # parser.add_argument('-l', '--label', type=str, required=True,
    #                     help='Specify if single or multi-label df.')
    parser.add_argument('--batch_size', type=str, help='mini-batch size used when iterating inside each file')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    # parser.add_argument('--val', type=str, default=None, help='Optional validation file (.pkl or .csv)')

    args = parser.parse_args()
    
    trdata      = args.trdata
    # label       = args.label
    batch_size  = args.batch_size
    epochs      = args.epochs
    lr          = args.lr
    # val         = args.val
    
    
    # Handle batch_size argument
    if batch_size is None:
        batch_size = None
        print("➡️ Running in FULL-DATA mode")
    else:
        batch_size = int(batch_size)
        print("➡️ Running in BATCH mode")
    
    # Default initializations
    le           = None
    num_labels   = None
    train_loader = None
    test_loader  = None
    
    #onfig     = load_cfg()
    
    config = Config('configurations')
    config = config.read("config.cfg")
    
    ROOT_DIR = config["ROOT_DIR"]
    FNAME    = config["FNAME"]
    LABEL    = config["LABEL"]
    
    params = Config('parameters')
    params = params.read("params.cfg")
    
    lr     = ast.literal_eval(params["LR"])
    epochs = ast.literal_eval(params["EPOCHS"])
    
    # ===============================
    # MODE 1: Full dataset (no batching)
    # ===============================
    if batch_size is None:
        if trdata is None:
            raise ValueError("Either --trdata (full file) or --batch_files (folder/files) must be provided")
        
        def prepare_dataset(trdata):
            df = pickle.load(open(trdata, "rb")) if trdata.endswith('.pkl') else pd.read_csv(trdata)
            feature_cols = [c for c in df.columns if c.startswith('X')]
            X = df[feature_cols].to_numpy(dtype=np.float32)
            y = df['Y'].astype(str).to_numpy()
        
            le = LabelEncoder()
            y_enc = le.fit_transform(y)
            num_labels = len(le.classes_)
            
            return X, y_enc, num_labels, le
        
        X, y_enc, num_labels, le = prepare_dataset(trdata)
        dataset = EncodedDataset(X, y_enc)
        
        def split_data(dataset): # Split data just for informative purposes, data is already properly split outside this script
            train_size = int(0.8 * len(dataset))
            test_size = len(dataset) - train_size
            train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
            
            train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=64)
            
            return train_loader, test_loader
        
        train_loader, test_loader = split_data(dataset)
        input_dim = X.shape[1]
    
    # ===============================
    # MODE 2: Batch-folder mode
    # ===============================
    else:
        trdata = trdata + '/' + FNAME
        
        def prepare_batches(trdata):
            print("Processing batches...")
            
            train_folder = Path(trdata, "train")
            test_folder  = Path(trdata, "test")
        
            train_batches = sorted(train_folder.glob("*.pkl"))
            test_batches  = sorted(test_folder.glob("*.pkl"))
        
            if not train_batches:
                raise FileNotFoundError(f"No .pkl batches found in {train_folder}")
            if not test_batches:
                print(f"⚠️ No test batches found in {test_folder}, training only.")
        
            # Collect all labels from all batches for consistent label encoding
            all_labels = []
            for bf in train_batches + test_batches:
                with open(bf, "rb") as f:
                    df_b = pickle.load(f)
                all_labels.extend(df_b['Y'].astype(str).tolist())
        
            le = LabelEncoder()
            le.fit(np.unique(all_labels))
            num_labels = len(le.classes_)
        
            # Derive input_dim from first training batch
            with open(train_batches[0], "rb") as f:
                df0 = pickle.load(f)
            feature_cols = [c for c in df0.columns if c.startswith('X')]
            input_dim = len(feature_cols)
        
            # Define loaders (lazy loading inside train_transformer)
            train_loader = None
            test_loader = None
            
            return train_batches, test_batches, num_labels, input_dim, train_loader, test_loader, le
        
        train_batches, test_batches, num_labels, input_dim, train_loader, test_loader, le = prepare_batches(trdata)
    # ===============================
    # MODEL INITIALIZATION
    # ===============================
    model = SimpleTransformerClassifier(input_dim=input_dim, num_labels=num_labels)   
    # Dilute the learning rate when batch
    # if batch_size is not None:
    #     lr = lr / 3
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    # ===============================
    # TRAINING
    # ===============================
    model, train_losses = train_transformer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        le=le,
        num_epochs=epochs,
        train_loader=train_loader,
        test_loader=test_loader,
        batch_files=(train_batches if batch_size is not None else None),
        batch_size=batch_size
    )
    
    # ===============================
    # SAVE MODEL
    # ===============================
    
    model_dir = Path(ROOT_DIR, 'results', 'models', LABEL, '16s_transformer', FNAME)
    os.makedirs(model_dir, exist_ok=True)
    
    plot_loss_curve(train_losses, save_path = model_dir / "loss_curve.jpg") # Save the train/val loss curve
    torch.save(model.state_dict(), model_dir / "model.pth")
    with open(model_dir / "label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)
    
    print(f"✅ Training finished, model saved in {model_dir.as_posix()}")


if __name__ == '__main__':
    main()
