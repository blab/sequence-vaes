import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader 
import sys
import json
from treetime.utils import datetime_from_numeric
from collections.abc import Iterable

sys.path.append("./VAE_standard")
from models import DNADataset, ALPHABET, SEQ_LENGTH, LATENT_DIM, VAE
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE as tsne
from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd

import bedford_code.models_bedford as bedford

from sklearn_extra.cluster import KMedoids
from scipy.optimize import minimize
from scipy.optimize._numdiff import approx_derivative
import pickle

from utils import minimize_curve, G_batched, compute_rho

BATCH_SIZE = 64

abspath = "."

# "data" directory is generated as shown in README.md file
dataset = DNADataset(f"{abspath}/data/training_spike.fasta")
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)

dset = ["training", "valid", "test"]
dset = dset[0]
abspath = "."
dataset = DNADataset(f"{abspath}/data/{dset}_spike.fasta")
new_dataset = np.array([dataset[x][0].numpy() for x in range(len(dataset))])
vals = np.array([dataset[x][1] for x in range(len(dataset))])
# labeling
metadata = pd.read_csv(f"{abspath}/data/all_data/all_metadata.tsv", sep="\t")
clade_labels = [metadata.loc[metadata.name == vals[i], "clade_membership"].values[0] for i in range(len(vals))]
dates = [metadata.loc[metadata.name == vals[i], "date"].values[0] for i in range(len(vals))]
dates = [datetime_from_numeric(x) for x in dates]

input_dim = len(ALPHABET) * SEQ_LENGTH
#STANDARD
vae_model = VAE(input_dim=input_dim, latent_dim=50).to(DEVICE)
vae_model.load_state_dict(torch.load("./VAE_standard/model_saves/standard_VAE_model_BEST.pth", weights_only=True, map_location=DEVICE))


vae_model.eval()

X = torch.tensor(new_dataset)
print("X shape")
print(new_dataset.shape)
print(X.shape)
# X = X.to(DEVICE)
X = X.view(X.size(0), -1).to(DEVICE)
pca = PCA(n_components=3, svd_solver="full")

recon = None
Z_mean = None
Z_embedded = None
scatterplot = None
with torch.no_grad():
    # STANDARD
    Z_mean, Z_logvar = vae_model.encoder.forward(X)
    recon = vae_model.decoder.forward(Z_mean)
    recon = recon.view(recon.shape[0], -1).cpu()
    Z_mean = Z_mean.cpu().numpy()
    Z_var = torch.exp(Z_logvar).cpu().numpy()

    pca.fit(Z_mean - np.mean(Z_mean))

converter = np.vectorize(lambda x: ALPHABET[int(x)])

genome = np.dot(new_dataset, np.arange(len(ALPHABET)))
genome = np.reshape(converter(genome.ravel()), genome.shape)
genome_recon = torch.argmax(recon.view(recon.shape[0], -1, 5), dim=-1).cpu().detach().numpy()
genome_recon = np.reshape(converter(genome_recon.ravel()), genome.shape)   
print(np.sum(np.not_equal(genome_recon, genome)) / genome.shape[0])

_, uniq_indices, uniq_inverse = np.unique(genome, axis=0, return_index=True, return_inverse=True)
print(uniq_indices.shape)

Z_embedded = pca.transform(Z_mean - np.mean(Z_mean))
variances = pca.explained_variance_ratio_
tot = np.sum(variances)
print(variances)
print(f"total variance: {tot}")

c = 50
kmedoids = KMedoids(n_clusters=c, random_state=1, init="k-medoids++").fit(Z_mean[uniq_indices, :])
kmed_index = [uniq_indices[x] for x in kmedoids.medoid_indices_]

mu = Z_mean[kmed_index,:]
mu_genome = genome[kmed_index,:]

pts = []
for i in range(c):
    for j in range(i+1, c):
        pts.append((i,j))

Z_mean = torch.tensor(Z_mean).to(DEVICE)
Z_var = torch.tensor(Z_var).to(DEVICE)

mu = Z_mean[kmed_index,:]
sigma = Z_var[kmed_index,:]
rho = compute_rho(mu)

print(c)
print(len(pts))
geo_losses = []
for (p1, p2) in pts:
    _, glosses = minimize_curve(Z_mean[p1,:], Z_mean[p2,:], mu, sigma, rho, lam=1, smooth=1e-4, eps=1e-13, k=20, n_reps=8000)
    geo_losses.append(glosses[-1])
    print(".", end="", flush=True)
print()

# with open("geo_losses.txt", "wb") as f:
#     pickle.dump(geo_losses, f)

# with open("geo_losses.txt", "rb") as f:
#     b = pickle.load(f)
#     print(b)
