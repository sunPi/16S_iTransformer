#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
16S RNA Transformer - FASTA Preprocessing (Full or Batch)
Author: jr453
"""

import gzip
import pandas as pd
import numpy as np
import os
from Bio import SeqIO
from itertools import islice
import argparse
from utils import *

# ========================
# 1. Load FASTA helpers
# ========================
def load_fasta_full(fasta_path):
    handle = gzip.open(fasta_path, "rt") if fasta_path.endswith(".gz") else open(fasta_path, "r")
    parser = SeqIO.parse(handle, "fasta")

    rows = []
    for rec in parser:
        parts = rec.description.split(" ", 1)
        accession = parts[0]
        taxonomy = parts[1].split(";") if len(parts) > 1 else ["Unclassified"]
        tax_levels = taxonomy + ["Unclassified"] * (7 - len(taxonomy))
        kingdom, phylum, clazz, order, family, genus, species = tax_levels[:7]
        rows.append({
            "SampleID": accession,
            "sequence": str(rec.seq),
            "kingdom": kingdom,
            "phylum": phylum,
            "class": clazz,
            "order": order,
            "family": family,
            "genus": genus,
            "species": species,
        })
    handle.close()
    return pd.DataFrame(rows)

def load_fasta_batches(fasta_path, batch_size, n_max=None):
    handle = gzip.open(fasta_path, "rt") if fasta_path.endswith(".gz") else open(fasta_path, "r")
    parser = SeqIO.parse(handle, "fasta")

    batch_idx, total = 0, 0
    while True:
        records = list(islice(parser, batch_size))
        if not records:
            break

        rows = []
        for rec in records:
            parts = rec.description.split(" ", 1)
            accession = parts[0]
            taxonomy = parts[1].split(";") if len(parts) > 1 else ["Unclassified"]
            tax_levels = taxonomy + ["Unclassified"] * (7 - len(taxonomy))
            kingdom, phylum, clazz, order, family, genus, species = tax_levels[:7]
            rows.append({
                "SampleID": accession,
                "sequence": str(rec.seq),
                "kingdom": kingdom,
                "phylum": phylum,
                "class": clazz,
                "order": order,
                "family": family,
                "genus": genus,
                "species": species,
            })
        yield pd.DataFrame(rows), batch_idx
        batch_idx += 1
        total += len(records)
        if n_max and total >= n_max:
            break
    handle.close()

# ========================
# 2. One-hot encoding
# ========================
def one_hot_encode(seq, max_len):
    mapping = {"A":0, "C":1, "G":2, "T":3, "N":4}
    arr = np.zeros((max_len, len(mapping)), dtype=np.float32)
    for i, base in enumerate(seq[:max_len]):
        idx = mapping.get(base.upper(), 4)
        arr[i, idx] = 1.0
    return arr

def encode_dataframe(df, max_len):
    encoded = np.stack([one_hot_encode(seq, max_len) for seq in df["sequence"]])
    return encoded

def build_feature_columns(max_len):
    alphabet = ("A","C","G","T","N")
    return [f"X{i}_{nuc}" for i in range(max_len) for nuc in alphabet]

# ========================
# 3. Save dataframes
# ========================
def save_df(df, batch_id, max_len, out_prefix):
    taxa_levels = ["kingdom", "phylum", "class", "order", "family", "genus", "species"]
    X = encode_dataframe(df, max_len)
    X_flat = X.reshape(X.shape[0], -1)
    feature_cols = build_feature_columns(max_len)

    for level in taxa_levels:
        df_sub = pd.DataFrame({
            "SampleID": df["SampleID"],
            "Y": df[level]
        })
        df_features = pd.DataFrame(X_flat, columns=feature_cols)
        df_full = pd.concat([df_sub, df_features], axis=1)
        out_file = f"{out_prefix}_{level}_batch{batch_id}.pkl"
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        df_full.to_pickle(out_file)
        print(f"💾 Saved batch {batch_id} for level {level}: {len(df_full)} rows")

# ========================
# 4. Main logic
# ========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="16S RNA Transformer - Preprocess FASTA (Full or Batches)")
    parser.add_argument('-f', '--fasta', type=str, required=True, help='Input FASTA file path.')
    parser.add_argument('-b', '--batch_size', type=int, default=None, help='Batch size for processing (if None, full mode).')
    parser.add_argument('-n', '--n_max', type=int, default=None, help='Max number of records.')
    parser.add_argument('-l', '--max_length', type=int, default=1500, help='Max sequence length.')
    args = parser.parse_args()

    config = load_cfg(os.path.dirname(os.path.abspath(__file__)) + "/../config.cfg")
    ROOT_DIR = config["ROOT_DIR"]
    fasta_path = os.path.join(ROOT_DIR, args.fasta)
    out_prefix = os.path.join(ROOT_DIR, "data/16S_RNA/singlelabel/silva")

    if args.batch_size:
        print(f"⚙️ Running in BATCH mode (batch size = {args.batch_size})")
        for df_batch, batch_id in load_fasta_batches(fasta_path, args.batch_size, args.n_max):
            save_df(df_batch, batch_id, args.max_length, out_prefix)
    else:
        print("⚙️ Running in FULL mode (processing entire FASTA)")
        df_full = load_fasta_full(fasta_path)
        save_df(df_full, 0, args.max_length, out_prefix)

    print("✅ Preprocessing complete.")
