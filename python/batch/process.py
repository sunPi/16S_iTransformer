import gzip
import pandas as pd
import numpy as np
import os
from Bio import SeqIO
import argparse
from itertools import islice

# ========================
# 1. Load SILVA FASTA in batches
# ========================
def load_silva_fasta_batches(fasta_path, batch_size, n_max=None):
    handle = gzip.open(fasta_path, "rt") if fasta_path.endswith(".gz") else open(fasta_path, "r")
    parser = SeqIO.parse(handle, "fasta")

    batch_num = 0
    total = 0

    while True:
        records = list(islice(parser, batch_size))
        if not records:
            break

        rows = []
        for rec in records:
            parts = rec.description.split(" ", 1)
            accession = parts[0]
            taxonomy = parts[1].split(";") if len(parts) > 1 else ["Unclassified"]

            # pad to 7 levels
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

        yield pd.DataFrame(rows), batch_num
        batch_num += 1
        total += len(records)

        if n_max and total >= n_max:
            break

    handle.close()

# ========================
# 2. One-hot encoding helpers
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

def build_feature_columns(max_len, alphabet=("A","C","G","T","N")):
    return [f"X{i}_{nuc}" for i in range(max_len) for nuc in alphabet]

# ========================
# 3. Save batches
# ========================
def save_batches(df, batch_id, max_len, out_prefix):
    taxa_levels = ["kingdom", "phylum", "class", "order", "family", "genus", "species"]

    X = encode_dataframe(df, max_len=max_len)
    X_flat = X.reshape(X.shape[0], -1)
    feature_cols = build_feature_columns(max_len=max_len)

    for level in taxa_levels:
        df_sub = pd.DataFrame({
            "SampleID": df["SampleID"],
            "Y": df[level]
        })
        df_features = pd.DataFrame(X_flat, columns=feature_cols)
        df_full = pd.concat([df_sub, df_features], axis=1)
        
        out_file    = f"{out_prefix}_{level}_batch{batch_id}.pkl"
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        df_full.to_pickle(out_file)

        print(f"💾 Saved batch {batch_id} for level {level}: {len(df_full)} rows -> {out_file}")

# ========================
# 4. Main
# ========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="16S RNA Transformer - Process in Batches")
    parser.add_argument('-f', '--fasta', type=str, required=True, help='Input fasta file (SILVA).')
    parser.add_argument('-b', '--batch_size', type=int, default=10000, help='Batch size for processing.')

    args = parser.parse_args()

    fasta_file = args.fasta
    batch_size = args.batch_size
    max_len = 1500
    n_max    = 1000000
    
    ROOT_DIR = "/home/jr453/Documents/Projects/Reem_16s_RNA_classification/"
    fasta_file = ROOT_DIR + "data/16S_RNA/SILVA_138.2_SSURef_NR99_tax_silva.fasta"
    
    print(f"🚀 Processing {fasta_file} in batches of {batch_size}")

    out_prefix = ROOT_DIR + "data/singlelabel/silva"
    for df_batch, batch_id in load_silva_fasta_batches(fasta_file, batch_size, n_max):
        save_batches(df_batch, batch_id, max_len, out_prefix)

    print("✅ Finished batch preprocessing")
