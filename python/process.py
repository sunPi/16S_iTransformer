import gzip
import pandas as pd
import numpy as np
import os
from Bio import SeqIO
import argparse
from utils import *

# ========================
# 1. Load SILVA FASTA with taxonomy
# ========================
def load_silva_fasta(fasta_path, n_max=None):
    records = []
    handle = gzip.open(fasta_path, "rt") if fasta_path.endswith(".gz") else open(fasta_path, "r")

    for i, rec in enumerate(SeqIO.parse(handle, "fasta")):
        parts = rec.description.split(" ", 1)
        accession = parts[0]
        taxonomy = parts[1].split(";") if len(parts) > 1 else ["Unclassified"]

        # pad to 7 levels
        tax_levels = taxonomy + ["Unclassified"] * (7 - len(taxonomy))
        kingdom, phylum, clazz, order, family, genus, species = tax_levels[:7]

        records.append({
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

        if n_max and i + 1 >= n_max:
            break

    handle.close()
    return pd.DataFrame(records)

# ========================
# 2. One-hot encode sequences
# ========================
def one_hot_encode(seq, max_len):
    mapping = {"A":0, "C":1, "G":2, "T":3, "N":4}
    arr = np.zeros((max_len, len(mapping)), dtype=np.float32)
    for i, base in enumerate(seq[:max_len]):
        idx = mapping.get(base.upper(), 4)
        arr[i, idx] = 1.0
    return arr  # shape [seq_len, 5]

def encode_dataframe(df, max_len):
    # produce 3D array: [samples, max_len, 5]
    encoded = np.stack([one_hot_encode(seq, max_len) for seq in df["sequence"]])
    return encoded

def build_feature_columns(max_len, alphabet=("A","C","G","T","N")):
    # e.g., X0_A, X0_C, X0_G, X0_T, X0_N, X1_A, ...
    return [f"X{i}_{nuc}" for i in range(max_len) for nuc in alphabet]

# ========================
# 3. Build per-taxa DFs
# ========================
def build_single_label_dfs(df, max_len, out_prefix="silva"):
    taxa_levels = ["kingdom", "phylum", "class", "order", "family", "genus", "species"]
    dfs = {}
    X = encode_dataframe(df, max_len=max_len)           # 3D array
    X_flat = X.reshape(X.shape[0], -1)                 # flatten to 2D [samples, max_len*5]
    feature_cols = build_feature_columns(max_len=max_len)

    for level in taxa_levels:
        df_sub = pd.DataFrame({
            "SampleID": df["SampleID"],
            "Y": df[level]
        })
        df_features = pd.DataFrame(X_flat, columns=feature_cols)
        df_full = pd.concat([df_sub, df_features], axis=1)

        dfs[level] = df_full

        # Save Pickle
        os.makedirs(os.path.dirname(f"{out_prefix}_{level}.pkl"), exist_ok=True)
        df_full.to_pickle(f"{out_prefix}_{level}.pkl")

        # Save CSV
        df_full.to_csv(f"{out_prefix}_{level}.csv", index=False)

        print(f"Saved {out_prefix}_{level}.pkl and .csv with {len(df_full)} rows")

    return dfs

# ========================
# 4. Build multi-label DF
# ========================
def build_multilabel_df(df, max_len, out_file_prefix="silva_multilabel"):
    X = encode_dataframe(df, max_len=max_len)
    X_flat = X.reshape(X.shape[0], -1)
    feature_cols = build_feature_columns(max_len=max_len)

    df_labels = df[["SampleID","kingdom","phylum","class","order","family","genus","species"]]
    df_full = pd.concat([df_labels.reset_index(drop=True), pd.DataFrame(X_flat, columns=feature_cols)], axis=1)

    # Save Pickle
    os.makedirs(os.path.dirname(f"{out_file_prefix}.pkl"), exist_ok=True)
    df_full.to_pickle(f"{out_file_prefix}.pkl")

    # Save CSV
    df_full.to_csv(f"{out_file_prefix}.csv", index=False)

    print(f"Saved {out_file_prefix}.pkl and .csv with {len(df_full)} rows")
    return df_full


# ========================
# 5. Main
# ========================
if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="16S RNA Transformer - Process")
    parser.add_argument('-f', '--fasta', type=str, required=True, help='File path for input data in fasta format.')
    
    # Parse arguments
    args = parser.parse_args()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config = load_cfg(os.path.dirname(script_dir) + "/config.cfg")

    ROOT_DIR = config["ROOT_DIR"]
    print(ROOT_DIR)

    # ROOT_DIR = "/home/jr453/Documents/Projects/Reem_16s_RNA_classification/"
    fasta_file = ROOT_DIR + "/" +  args.fasta
    # fasta_file = ROOT_DIR + "data/16S_RNA/SILVA_138.2_SSURef_NR99_tax_silva.fasta"
    
    
    n_max = None  # set to None to load all
    max_len = 1600
    
    print("Loading SILVA fasta...")
    df = load_silva_fasta(fasta_file, n_max=n_max)
    print(df.head())
    print(df.shape)
    
    # per-level
    out_prefix = ROOT_DIR + "/data/singlelabel/silva"
    os.makedirs(os.path.dirname(out_prefix), exist_ok=True)
    single_label_dfs = build_single_label_dfs(df, out_prefix=out_prefix, max_len=max_len)

    # multi-label
    out_prefix = ROOT_DIR + "/data/multilabel/silva_multilabel"
    os.makedirs(os.path.dirname(out_prefix), exist_ok=True)
    multilabel_df = build_multilabel_df(df, max_len=max_len, out_file_prefix=out_prefix)

    print("✅ Preprocessing finished")
