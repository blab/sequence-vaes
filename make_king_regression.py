import arviz as az
import pickle
import cloudpickle

import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.utils.data import DataLoader
import sys
import json

abspath = "./VAE_standard"

sys.path.append(abspath)
from models import DNADataset, ALPHABET, SEQ_LENGTH, LATENT_DIM, VAE
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE as tsne
from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd

import bedford_code.models_bedford as bedford
from treetime.utils import datetime_from_numeric
import pymc as pm
from collections.abc import Iterable
import altair as alt

BATCH_SIZE = 64
# "data" directory is generated as shown in README.md file
dataset = DNADataset(f"{abspath}/../data/training_spike.fasta")
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)

input_dim = len(ALPHABET) * SEQ_LENGTH
# input_dim = 29903 * 5
# input_dim = 29903

# BEDFORD
# vae_model = bedford.VAE(input_dim=len(bedford.ALPHABET) * bedford.SEQ_LENGTH, latent_dim=bedford.LATENT_DIM).to(DEVICE)
# vae_model.load_state_dict(torch.load(f"{abspath}/bedford_code/results_bedford/BEST_vae_ce_anneal.pth"))
#STANDARD
vae_model = VAE(input_dim=input_dim, latent_dim=50).to(DEVICE)
vae_model.load_state_dict(torch.load(f"{abspath}/model_saves/standard_VAE_model_BEST.pth", weights_only=True, map_location=DEVICE))

vae_model.eval()

dset = ["training", "valid", "test"]
dset = dset[0]
print(dset)
abspath = "."
dataset = DNADataset(f"{abspath}/data/{dset}_spike.fasta")
new_dataset = np.array([dataset[x][0].numpy() for x in range(len(dataset))])
vals = np.array([dataset[x][1] for x in range(len(dataset))])

def flatten(xs):
    for x in xs:
        if isinstance(x, Iterable) and not isinstance(x, (str, float)):
            yield from flatten(x)
        else:
            yield x

metadata = pd.read_csv(f"{abspath}/data/all_data/all_metadata.tsv", sep="\t")
clade_labels = [metadata.loc[metadata.name == vals[i], "clade_membership"].values[0] for i in range(len(vals))]
good_clade_labels = []
for c in clade_labels:
    if len(metadata[metadata.clade_membership == c]) > 5:
        good_clade_labels.append(c)
print(set(good_clade_labels))

# print(set(clade_labels))

# clusters = np.sort(np.array(list(set(good_clade_labels))))
clusters = np.sort(np.array(list(set(clade_labels))))
print(clusters)
get_clade = lambda x: [True if elem == x else False for elem in clade_labels]
indexes = tuple([np.arange(len(clade_labels))[get_clade(x)] for x in clusters])

ranges = np.concatenate(indexes)
X = torch.tensor(new_dataset[ranges,:,:])
# X = X.to(DEVICE)
X = X.view(X.size(0), -1).to(DEVICE)
print("X shape")
print(new_dataset.shape)
print(X.shape)

pca = PCA(n_components=3, svd_solver="full")
Z_mean = None
Z_embedded = None
with torch.no_grad():
    # STANDARD
    Z_mean, Z_logvar = vae_model.encoder.forward(X)
    recon = vae_model.decoder.forward(Z_mean)
    recon = recon.view(recon.shape[0], -1).cpu()
    Z_mean = Z_mean.cpu()
    Z_std = torch.exp(0.5 * Z_logvar).cpu()
    # BEDFORD
    # recon, Z_mean, Z_logvar = vae_model.forward(X)
    # recon = recon.cpu().numpy()
    # Z_mean = Z_mean.cpu().numpy()

    pca.fit(Z_mean)
    Z_embedded = pca.transform(Z_mean - torch.mean(Z_mean))
    variances = pca.explained_variance_ratio_
    tot = np.sum(variances)
    print("\n",variances)
    print(f"total variance: {tot}")

# labeling
dates = [metadata.loc[metadata.name == vals[i], "date"].values[0] for i in range(len(vals))]
dates = [datetime_from_numeric(x) for x in dates]

coords = [(x1,x2,x3,t,c) for (x1,x2,x3),t,c in zip(Z_embedded, dates,clade_labels)]
coords = pd.DataFrame(data=coords, columns=["dim0","dim1","dim2","time","clade"])

avg_coords = coords.groupby("time")[["dim0","dim1","dim2"]].median().resample("ME").median().dropna().reset_index()

ubound = len(avg_coords)

x_vals = np.linspace(0,ubound,num=len(avg_coords)).astype("float32")[:,np.newaxis]
# import theano.tensor as tt
def build_coords_model(dim, num_draws=2000):
    y_vals = avg_coords[dim].values.astype('float32')
    print(x_vals.shape, y_vals.shape)

    model = None
    with pm.Model() as model:
        # l = pm.HalfCauchy('l', beta=20)
        l = pm.Uniform('l', 0, 30)

        # Covariance function
        log_s2_f = pm.Uniform('log_s2_f', lower=-10, upper=5)
        s2_f = pm.Deterministic('s2_f', np.exp(log_s2_f))
        f_cov = s2_f * pm.gp.cov.ExpQuad(input_dim=1, ls=l)

        # Sigma = 1/lam
        s2_n = pm.HalfCauchy('s2_n', beta=5)

        gp = pm.gp.Latent(cov_func=f_cov)
        f = gp.prior("f",X=x_vals)

        df = 1 + pm.Gamma("df",alpha=2,beta=1)
        y_obs = pm.StudentT("y", mu=f, lam=1.0 / s2_n, nu=df, observed=y_vals)

        trace = pm.sample(draws=num_draws)
    return trace, gp, model


ls = ["dim0","dim1","dim2"]
# ls = ["dim2"]
N_draw = 6000

print(ls,"\n",N_draw)

ret_vals = [build_coords_model(d, num_draws=N_draw) for d in ls]

GPs = [x[1] for x in ret_vals]
idata = [x[0] for x in ret_vals]
models = [x[2] for x in ret_vals]

abspath = "."
dict_to_save = {x:(idata[i],GPs[i], models[i]) for i,x in enumerate(["dim0","dim1","dim2"])}
with open(f"{abspath}/king_regression_data.pkl","wb") as buff:
    cloudpickle.dump(dict_to_save, buff)

for n,d in zip(ls,idata):
    print(n)
    n_nonconverged = int(
        np.sum(az.rhat(d)[["l", "log_s2_f", "s2_n", "f_rotated_", "df"]].to_array() > 1.03).values
    )
    if n_nonconverged == 0:
        print("No Rhat values above 1.03, \N{check mark}")
    else:
        print(f"The MCMC chains for {n_nonconverged} RVs appear not to have converged.")
