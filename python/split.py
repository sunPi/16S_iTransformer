#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 10 21:07:38 2025

@author: jr453
"""

import pandas as pd
import os
from sklearn.model_selection import train_test_split
import argparse
from pathlib import Path
from collections import Counter
import math

# ========================
# 1. Functions
# ========================
def data_split(df):
    # Compute sizes for 80/20 split
    # train_size = int(0.8 * len(df))
    # test_size = len(df) - train_size

    def plot_histogram_classes(df):
        import matplotlib.pyplot as plt

        # Assuming your labels are in df['Y']
        class_counts = df['Y'].value_counts().sort_values(ascending=False)
        
        plt.figure(figsize=(10, 5))
        plt.bar(class_counts.index.astype(str), class_counts.values)
        plt.xticks(rotation=90)
        plt.xlabel("Class")
        plt.ylabel("Number of samples")
        plt.title("Class Distribution")
        plt.tight_layout()
        plt.show()  
        
    # plot_histogram_classes(df)
    def handle_rare_clases(df):  
        # Extract labels
        y = df["Y"]
        
        # Assuming your labels are in df['Y']
        counts = df['Y'].value_counts()
        
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
    
    df = handle_rare_clases(df)
    y = df['Y']
    # Split into train (80%) and test (20%)
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=y, random_state=42, shuffle=True)
    
    return train_df, test_df
def save_batches(df, batch_size=100, out_prefix="batch", outfolder="batches"):
    """
    Split a DataFrame into batches and save each as a .pkl file.

    Parameters
    ----------
    df : pandas.DataFrame
        The full dataset to split.
    batch_size : int
        Number of rows per batch.
    out_prefix : str
        Filename prefix for saved files.
    outfolder : str
        Directory where pickle files will be stored.
    """
    n_batches = math.ceil(len(df) / batch_size)

    for i in range(n_batches):
        start = i * batch_size
        end = start + batch_size
        batch_df = df.iloc[start:end]
        out_path = os.path.join(outfolder, f"{out_prefix}_batch_{i+1}.pkl")
        batch_df.to_pickle(out_path)
        
       #print(f"✅ Saved {out_path} ({len(batch_df)} rows)")

    print(f"\nTotal: {n_batches} batches saved.")

# ========================
# 2. Main
# ========================
if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="16S RNA Transformer - Split")
    parser.add_argument('-f', '--file', type=str, required=True, help='File path for the processed fasta files in either .csv or .pkl.')
    parser.add_argument('-b', '--batch_size', required=False, help='If set, processes the train data into batches of determined size', default=None)

    # Parse arguments
    args = parser.parse_args()
        
    # Input and output file paths
    #input_file = "/home/jr453/Documents/Projects/Reem_16s_RNA_classification/16S_iTransformer/data/16S_RNA/singlelabel/silva_species.pkl"
    #batch_size = 100
    input_file = args.file
     
    if args.batch_size == "None":
        batch_size = None
    else:
        batch_size = args.batch_size
        
    # Get filename of the input file
    fname = os.path.splitext(os.path.basename(input_file))[0]
    
    # Create names of the output folders
    output_folder_train = Path(os.path.dirname(input_file), fname, "train")
    output_folder_test = Path(os.path.dirname(input_file), fname, "test")
    
    # Create output folders
    os.makedirs(output_folder_train, exist_ok=True)
    os.makedirs(output_folder_test, exist_ok=True)
    
    # Create output file names
    output_file_train = Path(output_folder_train, ("train_" + os.path.basename(input_file)))
    output_file_test  = Path(output_folder_test, ("test_" + os.path.basename(input_file)))
    
    # Detect input file type
    file_ext = os.path.splitext(input_file)[1].lower()
    
    # Read based on the input file type
    if file_ext == ".csv":
        df = pd.read_csv(input_file)
    elif file_ext == ".pkl":
        df = pd.read_pickle(input_file)
    else:
        raise ValueError("Unsupported file format. Use .csv or .pkl")
    
    # Split the data into 80% Training data and 20% Testing data
    train_df, test_df = data_split(df)
    
    # Detect output file type
    out_ext = os.path.splitext(input_file)[1].lower()
    
    # Save either train/test datasets or save train to batches and test normally
    if batch_size is not None:
        output_folder_train = Path(os.path.dirname(input_file), fname, "train")
        save_batches(train_df, int(batch_size), "train", output_folder_train)
        test_df.to_pickle(output_file_test)
        
        print("✅ Train-Test split finished")
        print(f"Test data saved to {output_file_test}")
        
    else:
        if out_ext == ".csv":
            train_df.to_csv(output_file_train, index=False)
            test_df.to_csv(output_file_test, index=False)
        elif out_ext == ".pkl":
            train_df.to_pickle(output_file_train)
            test_df.to_pickle(output_file_test)
        else:
            raise ValueError("Unsupported output format. Use .csv or .pkl")
            
        print("✅ Train-Test split inished")
        print(f"Training data saved to {output_file_train}")
        print(f"Test data saved to {output_file_test}")