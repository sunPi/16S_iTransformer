#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 22:59:00 2025

@author: jr453
"""

from utils import load_cfg
import os
import argparse
import pandas as pd
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt

# Set up argument parser
parser = argparse.ArgumentParser(description="16S RNA Transformer - Evaluate")
parser.add_argument('-f', '--models_dir', type=str, required=True, help='File path to the models directory.')

# Parse arguments
args = parser.parse_args()

script_dir = os.path.dirname(os.path.abspath(__file__))
cfg_path   = os.path.dirname(script_dir) + "/config.cfg"
cfg_path   = "/home/jr453/Documents/Projects/Reem_16s_RNA_classification/16S_iTransformer/config.cfg"
config = load_cfg(cfg_path)

ROOT_DIR = config["ROOT_DIR"]

res_dir = Path(ROOT_DIR, 'results', 'models', 'single', '16s_transformer')

all_results = []
for dir_path in res_dir.glob("*/"):  # directly yields Path objects
    if dir_path.is_dir():
        print("Directory:", dir_path)
        
        res_path = Path(dir_path, 'metrics.csv')
        dname = os.path.basename(dir_path)
        
        metrics  = pd.read_csv(res_path)
        # Make sure metrics is a DataFrame
        metrics = metrics.copy()
        metrics.insert(0, "Experiment", dname)
        all_results.append(metrics)

# Concatenate all into one DataFrame
final_df = pd.concat(all_results, ignore_index=True)
final_df = final_df.sort_values(by="Accuracy", ascending=False).reset_index(drop=True)
print(final_df)

# Melt into long format
df_long = final_df.melt(
    id_vars="Experiment",
    value_vars=["Accuracy", "Precision", "Recall", "F1-score"],
    var_name="Metric",
    value_name="Score"
)

# Plot
plt.figure(figsize=(10, 6))
sns.barplot(
    data=df_long,
    x="Experiment",
    y="Score",
    hue="Metric",
    palette="Set2"
)
plt.xticks(rotation=45)
plt.title("Performance Metrics Across Experiments")
plt.tight_layout()
# Save the figure
plt.savefig(Path(res_dir, 'overall_barplot.png'), dpi=300, bbox_inches="tight")
print(f"Plot saved to: {res_dir}")

plt.show()


