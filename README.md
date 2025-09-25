# 16S_iTransformer

# 🧬 16S rRNA Taxonomy Classification with Transformers

This repository contains a modular pipeline for processing, training, and evaluating **Transformer-based models** on 16S rRNA sequence data. The pipeline supports both **single-label** and **multi-label** classification tasks and can be run end-to-end using a simple bash wrapper.  

---

## 📂 Pipeline Structure

The project consists of the following main scripts:

- **process.py** → Preprocesses FASTA input into one-hot encoded feature matrices (pickled `.pkl` or `.csv` format).  
- **split.py** → Splits data into train/test sets with stratification and handling of rare classes.  
- **train.py** → Trains a Transformer classifier on the processed data. Supports incremental learning and class expansion.  
- **evaluate.py** → Evaluates trained models on holdout or external data, reporting accuracy, precision, recall, F1, and visualizations.  
- **run_pipe** → A Bash wrapper (using `docopt`) that connects the above scripts for streamlined execution.  

---

## ⚙️ Installation

Create a clean conda environment and install dependencies:  

```bash
conda create -n 16s_rna python=3.11 -y
conda activate 16s_rna

# Install required packages
pip install torch scikit-learn pandas numpy matplotlib seaborn
