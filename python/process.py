import gzip
import pandas as pd
import numpy as np
import os
from Bio import SeqIO
import argparse
from utils import *
from pathlib import Path
from collections import Counter
import ast

# ========================
# 1. Functions
# ========================
# Loader Function
def load_silva_fasta(fasta_path, n_max=None):
    records = [] # Creates and empty list to store sequence records
    handle = gzip.open(fasta_path, "rt") if fasta_path.endswith(".gz") else open(fasta_path, "r")

    for i, rec in enumerate(SeqIO.parse(handle, "fasta")): # Loop over records in a fasta file and split data
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

# Sequence processing functions
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
def process_records(df, max_len, levels):
    X = encode_dataframe(df, max_len=max_len)
    X_flat = X.reshape(X.shape[0], -1)
    feature_cols = build_feature_columns(max_len=max_len)

    for level in levels:
        df_sub = pd.DataFrame({
            "SampleID": df["SampleID"].values,
            "Y": df[level].values
        }).reset_index(drop=True)

        df_features = pd.DataFrame(X_flat, columns=feature_cols).reset_index(drop=True)

        df_full = pd.concat([df_sub, df_features], axis=1)
        
        # sanity check
        if df_full.isna().any().any():
            print(f"⚠️ NaNs introduced at level {level} — likely index mismatch or malformed input")

        yield df_full, level

def iter_batches(df, batch_size=5000):
    for i in range(0, len(df), batch_size):
        yield df.iloc[i:i+batch_size]
        
def handle_rare_clases(df, verbose):
    if(verbose):
        print("Handling rare cases...")
    
    # Extract labels
    y = df["Y"]
    
    # Assuming your labels are in df['Y']
    counts = y.value_counts()
    
    # Count how many classes have fewer than 2 samples
    num_rare_classes = (counts < 2).sum()
    if num_rare_classes > 0:
        # Option 1: Remove classes with only 1 element per class
        counts = Counter(y)
        df = df[df['Y'].map(counts) >= 2]
        y = df['Y']
        
    else:
        # Option 2: Group them into 'Other'
        # Step 1: Group rare classes into 'Other'
        threshold = 2
        counts = y.value_counts()
        rare_classes = counts[counts < threshold].index
        df['label'] = df['Y'].replace(rare_classes, 'Other')
        y = df['label']
        
        # Step 2: Drop any classes (including 'Other') with < 2 samples
        try:
            counts = y.value_counts()
            valid_classes = counts[counts >= 2].index
            before = len(df)
            df = df[df['label'].isin(valid_classes)]
            y = df['label']
            after = len(df)
        
            if after < before:
                print(f"Dropped {before - after} rows from singleton classes.")
            if df.empty:
                raise ValueError("All rows were dropped after filtering singleton classes.")
        except Exception as e:
            print(f"⚠️ Warning while filtering singleton classes: {e}")
            print("Proceeding without filtering.")
            y = df['label']
    
    return df

# Dataframe builders
# Single-label DF
def build_single_label_dfs(df, max_len, out_prefix="silva", levels=None):
    taxa_levels = ["kingdom", "phylum", "class", "order", "family", "genus", "species"]

    # If no specific levels requested, process all
    if levels is None:
        levels = taxa_levels
    else:
        # Ensure it's a list even if user passes a single string
        if isinstance(levels, str):
            levels = [levels]
        # Validate
        levels = [lvl for lvl in levels if lvl in taxa_levels]

    dfs = {}
    X = encode_dataframe(df, max_len=max_len)
    X_flat = X.reshape(X.shape[0], -1)
    feature_cols = build_feature_columns(max_len=max_len)

    for level in levels:
        df_sub = pd.DataFrame({
            "SampleID": df["SampleID"],
            "Y": df[level]
        })
        df_features = pd.DataFrame(X_flat, columns=feature_cols)
        df_full = pd.concat([df_sub, df_features], axis=1)

        dfs[level] = df_full

        os.makedirs(os.path.dirname(f"{out_prefix}_{level}.pkl"), exist_ok=True)
        df_full.to_pickle(f"{out_prefix}_{level}.pkl")
        df_full.to_csv(f"{out_prefix}_{level}.csv", index=False)

        print(f"Saved {out_prefix}_{level}.pkl and .csv with {len(df_full)} rows")

    return dfs

# ========================
# 4. Build multi-label DF
# ========================
# Multi-label DF
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
    # return df_full

# Write out dataframes
def write_df(df, out_prefix):
    #os.makedirs(os.path.dirname(), exist_ok=True)
    df.to_pickle(f"{out_prefix}.pkl")
    df.to_csv(f"{out_prefix}.csv", index=False) 

# ========================
# 2. Main
# ========================
if __name__ == "__main__":
    ########### Set up argument parser
    parser = argparse.ArgumentParser(description="16S RNA Transformer - Process")
    parser.add_argument('-f', '--fasta', type=str, required=True, help='File path for input data in fasta format.')
    parser.add_argument('-n', '--n_max', type=int, required=True, default=None, help='Maximum number of sequences to process.')
    parser.add_argument('-l', '--max_length', type=int, required=True, default=1600, help='Maximum length of sequences to process.')
    parser.add_argument('-c', '--levels', type=str, help='Specify a subset of taxonomy to process.')
    parser.add_argument('-b', '--batch_size', type=str, help='Specify the size of batches to process the data on.')
    
    ########### Parse arguments
    args = parser.parse_args()
    # script_dir = "/home/jr453/Documents/Projects/Reem_16s_RNA_classification/16S_iTransformer/python"
    # ROOT_DIR = "/home/jr453/Documents/Projects/Reem_16s_RNA_classification/"
    # fasta_file = ROOT_DIR + "data/16S_RNA/SILVA_138.2_SSURef_NR99_tax_silva.fasta"
    
    config = load_cfg()
    ROOT_DIR = config["ROOT_DIR"]
    VERBOSE  = ast.literal_eval((config["VERBOSE"]))
    config["LABEL"] = "singlelabel"
    update_config(config)
    
    print(VERBOSE)
    
    if(VERBOSE):
        print(ROOT_DIR)
    
    fasta_file = ROOT_DIR + "/" +  args.fasta
    n_max   = args.n_max
    max_len = args.max_length
    levels  = args.levels
    batch_size = args.batch_size

    if levels == "None":
        levels = None
    if batch_size == "None":
        batch_size = None
    else:
        batch_size = int(batch_size)
        
    ########### Load In Fasta File
    print("Loading SILVA fasta...")
    df = load_silva_fasta(fasta_file, n_max=n_max)
    
    if(VERBOSE):
        print(df.head())
        print(df.shape)  
    
    # per-level
    out_prefix = ROOT_DIR + "/data/16S_RNA/singlelabel/silva"
    os.makedirs(os.path.dirname(out_prefix), exist_ok=True)
    
    taxa_levels = ["kingdom", "phylum", "class", "order", "family", "genus", "species"]
    
    ########### Ensure taxonomy levels are handled correctly
    # If no specific levels requested, process all
    if levels is None: # If user doest not specify to process a specific level
        levels = taxa_levels
        config["LEVELS"] = levels
        update_config(config)
        
    else:
        # Ensure it's a list even if user passes a single string
        if isinstance(levels, str):
            levels = [levels]
            
        # Validate
        invalid = [lvl for lvl in levels if lvl not in taxa_levels]
        if invalid:
            raise ValueError(f"Inappropriate level variable(s): {invalid}. Did you misspell a taxonomy level?")

        levels = [lvl for lvl in levels if lvl in taxa_levels] # If a user requests only one level
        config["LEVELS"] = levels
        update_config(config)
    
    ########### Process Fasta File and Build single/multi-label DF
    if batch_size is not None: # Process in batches
        def process_batches(df, batch_size, max_len, levels, ROOT_DIR, verbose):
            for i, batch in enumerate(iter_batches(df, batch_size), 1):
                for pbatch, level in process_records(batch, max_len, levels):  
                    if pbatch.isna().any().any():
                        raise ValueError(f"NaN found in batch {i}. Please check your data!")
                    if(verbose):
                        print(f"Processing {level} batch {i} ({len(batch)} rows)")
                        
                    out_prefix = Path(ROOT_DIR, f"{level}") 
                    out_prefix = Path(ROOT_DIR, 'data', '16S_RNA', 'singlelabel', 'batches', f'batch_{i}')
                    os.makedirs(os.path.dirname(out_prefix), exist_ok=True)
                    
                    pbatch = handle_rare_clases(pbatch, VERBOSE)
                    pbatch.to_pickle(f"{out_prefix}_{level}.pkl")
                    if(verbose):
                        print(f"Saved {out_prefix}_{level}.pkl and .csv with {len(pbatch)} rows")
        process_batches(df, batch_size, max_len, levels, ROOT_DIR, VERBOSE)
                
    else: # Processes full dataframes
        def process_df(df, max_len, levels, ROOT_DIR):
            for pdf, level in process_records(df, max_len, levels):  
                if(VERBOSE):
                    print(f"Processing {level}...")
                out_prefix = Path(ROOT_DIR, 'data', '16S_RNA', 'singlelabel', 'silva')
                out_prefix = f'{out_prefix}_{level}'
                os.makedirs(os.path.dirname(out_prefix), exist_ok=True)
                
                pdf = handle_rare_clases(pdf, VERBOSE)
                write_df(pdf, out_prefix)
                if(VERBOSE):
                    print(f"Saved {out_prefix}_{level}.pkl and .csv with {len(pdf)} rows")
                
            # multi-label
            out_prefix = ROOT_DIR + "/data/16S_RNA/multilabel/silva_multilabel"
            os.makedirs(os.path.dirname(out_prefix), exist_ok=True)
            
            build_multilabel_df(df, max_len=max_len, out_file_prefix=out_prefix)
        process_df(df, max_len, levels, ROOT_DIR, VERBOSE)
        
    print("✅ Data processing finished")