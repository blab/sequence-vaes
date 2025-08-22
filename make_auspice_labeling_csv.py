import os
import torch
import numpy as np
import pandas as pd
from VAE_standard.models import DNADataset
import numpy as np
from matplotlib import pyplot as plt

def make_csv_from_df(strain_names, secret_names, file_name, lab="secret",color=True):
    if not os.path.isfile(file_name):
        open(file_name, "x")

    with open(file_name, "w") as f:
        if color:
            f.write(f"strain,{lab},{lab}__colour\n")
            cmap = plt.get_cmap("gist_ncar")
            clusters = np.sort(np.array(list(set(secret_names))))

            test = [cmap(x)[:3] for x in np.arange(len(clusters)) / len(clusters)]
            test = [(int(x*255), int(y*255), int(z*255)) for (x,y,z) in test]
            test = ["#%02x%02x%02x" % x for x in test]

            color_dict = {c:x for c,x in zip(clusters, test)}
            for n,c in zip(strain_names, secret_names):
                col = color_dict[c]
                f.write(n + "," + c + "," + col + "\n")

        else:
            f.write(f"strain,{lab}\n")
            for n,c in zip(strain_names, secret_names):
                f.write(n + "," + c + "\n")

dsets = ["valid","training","test"]
abspath = "."

for dset in dsets:
    print(dset)
    dataset = DNADataset(f"{abspath}/data/{dset}_spike.fasta")
    new_dataset = np.array([dataset[x][0].numpy() for x in range(len(dataset))])
    vals = np.array([dataset[x][1] for x in range(len(dataset))])

    metadata = pd.read_csv(f"{abspath}/data/all_data/all_metadata.tsv", sep="\t")
    clade_labels = [metadata.loc[metadata.name == vals[i], "clade_membership"].values[0] for i in range(len(vals))]

    make_csv_from_df(vals, clade_labels, f"{dset}_labels.csv", lab=f"{dset}")   
