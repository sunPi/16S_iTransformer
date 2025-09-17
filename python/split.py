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

# Set up argument parser
parser = argparse.ArgumentParser(description="16S RNA Transformer - Split")
parser.add_argument('-f', '--file', type=str, required=True, help='File path for the processed fasta files in either .csv or .pkl.')

# Parse arguments
args = parser.parse_args()
    
# Input and output file paths
# input_file = "/home/jr453/Documents/Projects/Reem_16s_RNA_classification/data/singlelabel/silva_kingdom.csv"
input_file = args.file

fname = os.path.splitext(os.path.basename(input_file))[0]

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

# # Sample 20% of rows (without replacement by default)
# sampled_df = df.sample(frac=0.2, random_state=42, replace=False)

# Compute sizes for 80/20 split
train_size = int(0.8 * len(df))
test_size = len(df) - train_size

# Extract labels
y = df["Y"]

# Group classes with count < 2 into 'other' 
threshold = 2
counts = y.value_counts()
rare_classes = counts[counts < threshold].index
df['label'] = df['Y'].replace(rare_classes, 'Other')
y = df['label']

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

print(f"Training data saved to {output_file_train}")
print(f"Test data saved to {output_file_test}")