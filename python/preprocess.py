from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

import os
import re


#ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.getcwd()

in_fa = "data/16S_seq_nogap.fasta"
out_fa = "results/silva_cleaned.fasta"
min_len = 1200   # set to 400 if trimming to V3-V4 or to None to keep all
max_len = 1600

def clean_seq(s):
    s = s.replace(".", "").replace("-", "")      # remove alignment chars
    s = s.upper().replace("U", "T")              # RNA -> DNA
    s = re.sub(r"[^ACGT]", "N", s)               # non-ACGT -> N
    return s

out_recs = []
lengths = []
for rec in SeqIO.parse(in_fa, "fasta"):
    seq = str(rec.seq)
    seq = clean_seq(seq)
    L = len(seq)
    # if (min_len is not None and L < min_len) or (max_len is not None and L > max_len):
    #     # skip or you can keep; here we skip
    #     continue
    newrec = SeqRecord(Seq(seq), id=rec.id, description=rec.description)
    out_recs.append(newrec)
    lengths.append((rec.id, L))

SeqIO.write(out_recs, out_fa, "fasta")

# write lengths metadata
with open("results/silva_lengths.tsv", "w") as o:
    o.write("id\tlen\n")
    for r,L in lengths:
        o.write(f"{r}\t{L}\n")

print(f"Wrote {len(out_recs)} cleaned records to {out_fa}")
