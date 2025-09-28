# 16S_iTransformer

# 🧬 16S rRNA Taxonomy Classification with Transformers

This repository contains a modular pipeline for processing, training, and evaluating **Transformer-based models** on 16S rRNA sequence data. The pipeline supports both **single-label** and **multi-label** classification tasks and can be run end-to-end using a simple bash wrapper.  

---

## 📂 Pipeline Structure

The project consists of the following main scripts:

- **process.py** → Preprocesses FASTA input into one-hot encoded feature matrices (pickled `.pkl` or `.csv` format).  
- **split.py** → Splits data into train/test sets with stratification and handling of rare classes.  
- **train_transformer.py** → Trains a Transformer classifier on the processed data. Supports incremental learning and class expansion.
- **train_XGB.py** → Trains an eXtreme Gradient Booster classifier on the processed data. Used for validation/comparison.
- **train_grapple.py** → Trains a graphical model (GraPPLe) on the processed data. Not implemented yet fully!
- **evaluate.py** → Evaluates trained models on holdout or external data, reporting accuracy, precision, recall, F1, and visualizations.
- **plot_performance** → Collates and plots the data across taxa, if only analysing one taxa (e.g. species) no need to use this.
- **train_increment.py**  → Used for fine-tuning a pre-trained transformer model, either using already seen classes or adding new classes (i.e. class expansion).
- **run_pipe** → A Bash wrapper (using `docopt`) that connects the above scripts for streamlined execution.  

---

## ⚙️ Installation

Create a clean conda environment and install dependencies:  

```bash
conda create env -f env.yaml 
conda activate reem16s

# Install PyTorch (optional)
This pipeline relies on successfull installation of PyTorch modules with GPU support (together with the correct Nvidia CUDA version).
In case the PyTorch packages fail to install, please follow instructions from https://pytorch.org/get-started/locally/

```

---
## 🚀 Usage

The pipeline is controlled via the run_pipe script, which provides a command-line interface powered by docopt.

```bash
General Syntax:
./run_pipe <command> [options]

```bash
Minimal working example:

1. Download the SILVA training data (~500.000 samples) and uncompress it from:

https://www.arb-silva.de/fileadmin/silva_databases/current/Exports/SILVA_138.2_SSURef_NR99_tax_silva.fasta.gz

2. Put it inside your repository under data/16S_RNA/

3. Run the pipeline using your favorite terminal. In Linux use bash like this:

cd 16S_iTransformer

./BASH/run_pipe -f data/16S_RNA/SILVA_138.2_SSURef_NR99_tax_silva.fasta -n 5000 -l 100 -t 'itransformer'

Link to binder demo
https://hub.2i2c.mybinder.org/user/sunpi-16s_itransformer-fwt8lubs/lab/tree/demo.ipynb



