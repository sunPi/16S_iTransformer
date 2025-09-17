import glob
import pandas as pd

ROOT_DIR = "/home/jr453/Documents/Projects/Reem_16s_RNA_classification/"

# Get all batch files
batch_files = sorted(glob.glob(ROOT_DIR + "data/singlelabel/kingdom/*kingdom*"))

n_train = int(len(batch_files) * 0.8)

# Decide split
train_batches = batch_files[:n_train]   # first 80 batches for training
test_batches  = batch_files[n_train:]   # remaining for testing

print("Train:", len(train_batches), "batches")
print("Test:", len(test_batches), "batches")
