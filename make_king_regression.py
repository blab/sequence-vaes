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
import utils

BATCH_SIZE = 64
dset = ["training", "valid", "test"]
dset = dset[0]
abspath = "."

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)

# LOAD DATA
data_keys, data_dict = utils.get_data_dict(dset, abspath)
print(data_keys)
new_dataset = data_dict["new_dataset"]
vals = data_dict["vals"]
metadata = data_dict["metadata"]
clade_labels = data_dict["clade_labels"]
collection_dates = data_dict["collection_dates"]
indexes = data_dict["indexes"]
pairs = data_dict["pairs"]
get_parents_dict = data_dict["get_parents_dict"]


# LOAD MODEL
input_dim = len(ALPHABET) * SEQ_LENGTH
vae_model = VAE(input_dim=input_dim, latent_dim=50, non_linear_activation=nn.Softplus(beta=1.0)).to(DEVICE)
vae_model.load_state_dict(torch.load("./VAE_standard/model_saves/standard_VAE_model_BEST.pth", weights_only=True, map_location=DEVICE))
vae_model.eval()

# EVAL DATA
Z_mean, Z_logvar, recon, genome, genome_recon = utils.model_eval(vae_model, new_dataset, model_type="STANDARD")

# PCA REDUCE
pca = PCA(n_components=4, svd_solver="full")
pca.fit(Z_mean - np.mean(Z_mean))
Z_embedded = pca.transform(Z_mean - np.mean(Z_mean))
variances = pca.explained_variance_ratio_
tot = np.sum(variances)
print(variances)
print(f"total variance: {tot}")

# GP regression, sampling by month
# TIME vs. DIM data creation
metadata = pd.read_csv(f"{abspath}/data/all_data/all_metadata.tsv", sep="\t")
dates = [metadata.loc[metadata.name == vals[i], "date"].values[0] for i in range(len(vals))]
dates = [datetime_from_numeric(x) for x in dates]
coords = [(x1,x2,x3,x4,t,c) for (x1,x2,x3,x4),t,c in zip(Z_embedded, dates,clade_labels)]
coords = pd.DataFrame(data=coords, columns=["dim0","dim1","dim2","dim3","time","clade"])

# SAMPLE BY MONTH
avg_coords = coords.groupby("time")[["dim0","dim1","dim2","dim3"]].median().resample("ME").median().dropna().reset_index()
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


ls = ["dim0","dim1","dim2", "dim3"]
# ls = ["dim3"]
N_draw = 6000

print(ls,"\n",N_draw)

ret_vals = [build_coords_model(d, num_draws=N_draw) for d in ls]

idata = [x[0] for x in ret_vals]
GPs = [x[1] for x in ret_vals]
models = [x[2] for x in ret_vals]

abspath = "."
dict_to_save = {x:(idata[i],GPs[i], models[i]) for i,x in enumerate(ls)}
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
