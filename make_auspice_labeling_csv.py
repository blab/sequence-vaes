import os
import torch
import pandas as pd
from VAE_standard.models import DNADataset
import numpy as np

def make_csv_from_df(strain_names, secret_names, file_name):
    if not os.path.isfile(file_name):
        open(file_name, "x")

    with open(file_name, "w") as f:
        f.write("strain,secret\n")
        for n,c in zip(strain_names, secret_names):
            f.write(n + "," + c + "\n")

dsets = ["valid","training","test"]
abspath = "."

for dset in dsets:
    dataset = DNADataset(f"{abspath}/data/{dset}_spike.fasta")
    new_dataset = np.array([dataset[x][0].numpy() for x in range(len(dataset))])
    vals = np.array([dataset[x][1] for x in range(len(dataset))])

    metadata = pd.read_csv(f"{abspath}/data/all_data/all_metadata.tsv", sep="\t")
    clade_labels = [metadata.loc[metadata.name == vals[i], "clade_membership"].values[0] for i in range(len(vals))]

    make_csv_from_df(vals, clade_labels, f"{dset}_labels.csv")   
