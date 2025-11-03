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
from collections import defaultdict
import pickle
from utils import *
import re
import ast
import random
import math 
import shutil

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

# ========================
# 2. Main
# ========================
if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="16S RNA Transformer - Split")
    parser.add_argument('-f', '--file', type=str, required=True, help='File path for the processed fasta files in either .csv or .pkl.')
    parser.add_argument('-b', '--batch', type=str, required=False, help='If set, processes the train data into batches of determined size', default=False)

    # Parse arguments
    args = parser.parse_args()
        
    # Input and output file paths
    # input_file = "/home/jr453/Documents/Projects/Reem_16s_RNA_classification/16S_iTransformer/data/16S_RNA/singlelabel/silva_species.pkl"
    # script_dir = "/home/jr453/Documents/Projects/Reem_16s_RNA_classification/16S_iTransformer/python"
    # batch      = True
    
    input_file = args.file
    batch      = args.batch
    config     = load_cfg()
    
    if batch == "True":
        batch = True
    else:
        batch = False
    
    ROOT_DIR = config["ROOT_DIR"]
    taxa_levels = ast.literal_eval(config["LEVELS"])
        
    if batch:
        # input_file = Path("/home/jr453/Documents/Projects/Reem_16s_RNA_classification/16S_iTransformer/data/16S_RNA/singlelabel/batches/")
        def split_batches(input_file):
            # Collect all batch files
            batch_files = sorted(Path(input_file).glob("*.pkl"))
                  
            # Dictionary to hold grouped batch files by taxonomy level        
            batch_groups = defaultdict(list)
            
            for file_path in batch_files:
                fname = os.path.basename(file_path)
                match = re.search(r"batch_\d+_([a-zA-Z]+)\.pkl", fname)
                if match:
                    level = match.group(1)
                    batch_groups[level].append(file_path)
    
            for level in taxa_levels:
                nbatch = len(batch_groups[level]) # Get amount of batches
                random.shuffle(batch_groups[level]) # Shuffle them
                training_partition = math.ceil(nbatch * 0.8)
                testing_partition  = nbatch - training_partition
    
                train_data = batch_groups[level][0:training_partition]
                test_data  = batch_groups[level][-testing_partition:]
                
                config["FNAME"] = f'silva_{level}'
                update_config(config)
                
                output_folder_train = Path(os.path.dirname(input_file), 'batches', f'silva_{level}', "train")
                output_folder_test  = Path(os.path.dirname(input_file), 'batches', f'silva_{level}', "test")
                
                output_folder_train.mkdir(parents=True, exist_ok=True)
                output_folder_test.mkdir(parents=True, exist_ok=True)
    
                # Move files
                for f in train_data:
                    try:
                        shutil.move(f, output_folder_train / Path(f).name)
                    except Exception as e:
                        print(f"❌ Failed to move {f} → {output_folder_train}: {e}")
                
                output_folder_test_batches = output_folder_test / 'batches'
                output_folder_test_batches.mkdir(parents=True, exist_ok=True)
                
                tdfs = []
                for f in test_data:
                    try:
                        # Load and merge all test batch files
                        tdf = pd.read_pickle(f)
                        tdfs.append(tdf)   
                        shutil.move(f, output_folder_test_batches / Path(f).name)
                        #f.unlink()
                        
                    except Exception as e:
                        print(f"❌ Failed to merge test batches into a dataframe!")
                        
                # Combine into one DataFrame
                test_df = pd.concat(tdfs, ignore_index=True)

                # Save merged test set
                test_df.to_pickle(output_folder_test / f'test_silva_{level}.pkl')
                
                print(f"Merged test set saved as {output_folder_test / f'test_silva_{level}.pkl'}")
                print("✅ Train-Test split finished")
                
        split_batches(input_file)
    else:
        def split_dataframe(input_file):
            fname = os.path.splitext(os.path.basename(input_file))[0]
            config["FNAME"] = fname
            update_config(config)
            
            output_folder_train = Path(os.path.dirname(input_file), fname, "train")
            output_folder_test = Path(os.path.dirname(input_file), fname, "test")
            
            os.makedirs(output_folder_train, exist_ok=True)
            os.makedirs(output_folder_test, exist_ok=True)
            
            output_file_train = Path(output_folder_train, ("train_" + os.path.basename(input_file)))
            output_file_test  = Path(output_folder_test, ("test_" + os.path.basename(input_file)))
            
            # Detect input file type
            file_ext = os.path.splitext(input_file)[1].lower()
            
            if file_ext == ".csv":
                df = pd.read_csv(input_file)
            elif file_ext == ".pkl":
                df = pd.read_pickle(input_file)
            else:
                raise ValueError("Unsupported file format. Use .csv or .pkl")
            
            # Extract labels
            y = df["Y"]
    
            # Split into train (80%) and test (20%)
            train_df, test_df = train_test_split(df, test_size=0.2, stratify=y, random_state=42, shuffle=True)
            
            # Detect output file type
            out_ext = os.path.splitext(input_file)[1].lower()
            
            if out_ext == ".csv":
                train_df.to_csv(output_file_train, index=False)
                test_df.to_csv(output_file_test, index=False)
            elif out_ext == ".pkl":
                train_df.to_pickle(output_file_train)
                test_df.to_pickle(output_file_test)
            else:
                raise ValueError("Unsupported output format. Use .csv or .pkl")
            
            print("✅ Train-Test split Finished")
            print(f"Training data saved to {output_file_train}")
            print(f"Test data saved to {output_file_test}")
        
        split_dataframe(input_file)
        
        