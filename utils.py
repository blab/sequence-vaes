import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from VAE_standard.models import DNADataset, ALPHABET
import numpy as np
from matplotlib import pyplot as plt

from treetime.utils import datetime_from_numeric
from collections.abc import Iterable
import pandas as pd

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# HELPER FUNCTIONS
def make_csv_from_df(strain_names, secret_names, file_name, lab="secret",color=True):
    """
    HELPER function for make_auspice_labeling_csv
    """
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


# minimizer functions for geodesic ------------------------------
# Pytorch implementation for speed
def compute_rho(mu, eps=1e-12):
    """rho = max_i min_{j!=i} ||mu_i - mu_j||_2, vectorized."""
    diff = mu[:, None, :] - mu[None, :, :]  # (C,C,D)
    d = torch.sqrt(torch.sum(diff * diff, dim=-1)) # (C,C)
    d.fill_diagonal_(torch.inf)
    rho, _ = torch.min(d, dim=1)
    rho, _ = torch.max(rho[None,:] + eps, dim=-1)
    return rho

def dist_func(z, mu, sigma_inv): 
    diff = z[:, None, :] - mu[None, :, :] # (b, 1, dim) - (1, c, dim) = (b, c, dim)

    sigma_inv_mat = torch.diag_embed(sigma_inv) # (c, dim, dim)
    dist_sq = torch.einsum("bci, cij, bcj -> bc", diff.float(), sigma_inv_mat.float(), diff.float())

    # dist_sq = torch.square(torch.sum(diff * sigma_inv[None, :, :] * diff, dim=-1)) # (b, c, dim) * (1, c, dim) * (b, c, dim)
    return dist_sq

def omega(z, mu, sigma_inv, rho):
    dists_sq = dist_func(z, mu, sigma_inv)
    return torch.exp(-dists_sq / torch.square(rho))

def G_batched(z, mu, sigma_var, rho, lam, tau=0, eps=1e-12):
    sigma_inv = 1.0 / (sigma_var + eps) # (c, dim)
    omegas = omega(z, mu, sigma_inv, rho) # (b,c)
    summand = torch.matmul(omegas.double(), sigma_inv.double()) # (b,dim)
    summand = summand + lam * torch.exp(-1.0 * tau * torch.sum(torch.square(z), dim=-1))[:,None]
    return summand

def _curve_weights(k):
    w = torch.full(size=(1,k), fill_value=1.0 / (k - 1)).to(DEVICE)
    w[0] *= 0.5; w[-1] *= 0.5
    return w

def minimize_curve(z0, z1, mu, sigma_var, rho,
                   k=101, lam=0.0, tau=0.0,
                   smooth1=0.0, smooth2=0.0, seed=None, eps=1e-12, jac=None, DEVICE=DEVICE, n_reps=8000, jitter=5):
    """
    Minimize sum_i w_i * [1 / det G(z_i)] with fixed endpoints.
    - z0, z1: (D,)
    - mu: (C,D)
    - sigma_var: (C,D) diagonal variances of Σ_c
    - rho: float
    - k: number of discretization points (len(t)=k)
    - smooth: weight on sum ||Z[i+1]-Z[i]||^2 (optional)
    Returns (Z_opt, losses)
    """
    z0 = z0.to(DEVICE)
    z1 = z1.to(DEVICE)
    mu = mu.to(DEVICE)
    sigma_var = sigma_var.to(DEVICE)
    rho = rho.to(DEVICE)
    
    D = z0.shape[0]
    t = torch.linspace(0.0, 1.0, k)[:, None].to(DEVICE)
    Z0 = (1 - t) * z0[None, :] + t * z1[None, :]
    if k > 2:
        Z0[1:-1] += jitter * torch.randn((k-2, D)).to(DEVICE) # jitter
    w = _curve_weights(k)

    def pack(Z):  # (k,D) -> ((k-2)*D,)
        return torch.ravel(Z[1:-1])

    def unpack(x):  # ((k-2)*D,) -> (k,D) with fixed endpoints
        Z = torch.empty((k,D), dtype=float).to(DEVICE)
        Z[0] = z0
        Z[-1] = z1
        if k > 2:
            Z[1:-1] = x.view(k-2, D)
        return Z

    def objective(x):
        Z = unpack(x)
        Gd = G_batched(Z, mu, sigma_var, rho, lam=lam, tau=tau, eps=eps)   # (k,D)
        # 1 / sqrt(det(G)) computed stably: exp(-sum_d log Gd)
        inv_det = torch.exp(-0.5 * torch.sum(torch.log(Gd + eps), dim=1))        # (k,)
        val = torch.sum(w * inv_det)
        if smooth1 > 0.0 or smooth2 > 0.0:
            num_pts = Z.shape[0] - 1           

            diffs = Z[1:] - Z[:-1]

            # l2_dists = F.softmax(torch.sum(diffs, dim=-1), dim=0)
            # target = torch.full((1,num_pts), 1/num_pts).to(DEVICE)

            val += smooth1 * torch.sum(diffs * diffs)
            # val += smooth1 * torch.sum(torch.square(l2_dists - target)) + smooth2 * torch.sum(diffs * diffs)
        return val

    list_params = []
    params = pack(Z0).to(DEVICE)
    params.requires_grad_()
    optimizer = torch.optim.Adam([params], lr=1e-3)

    losses = []
    
    for i in range(n_reps):
        optimizer.zero_grad()
        loss = objective(params)
        loss.backward()
        optimizer.step()
        list_params.append(params.detach().clone())
        losses.append(loss.item())

    # disp_x = range(len(losses) // 10)
    # disp_y = [losses[x * 10] for x in range(len(losses) // 10)]
    # plt.plot(disp_x, disp_y)
    # plt.show()

    return unpack(params), losses

def path_cost(path, mu, sigma_var, rho, eps=1e-12, DEVICE=DEVICE, lam=0.0, tau=0.0):
    """
    path: (N, d) vector of N points that are d-dimensional

    returns: cumulative path cost wrt geodesic metric defined above (see minimize_curve)
    """
    Gd = G_batched(path, mu, sigma_var, rho, lam=lam, tau=tau, eps=eps)   # (k,D)
    inv_det = torch.exp(-0.5 * torch.sum(torch.log(Gd + eps), dim=1))        # (k,)
    val = torch.sum(inv_det)
    return val

def l2_path_len(geopath):
    """
    path: (N, d) vector of N points that are d-dimensional

    returns: l2 path length
    """
    l2_dists = np.sqrt(np.sum(np.square(geopath[1:,:] - geopath[:-1,:]), axis=-1))
    return np.sum(l2_dists)
    
def cos_sim(geopath):
    """
    path: (N, d) vector of N points that are d-dimensional

    returns: sum of angular differences across path
    """
    l2_norms = np.sqrt(np.sum(np.square(geopath), axis=-1))[:,None]
    unit_vec_path = np.divide(geopath, l2_norms)
    ang_dists = np.diagonal(np.dot(unit_vec_path[1:,:], unit_vec_path[:-1,:].T))
    ang_dists = np.sum(np.arccos(ang_dists))
    return ang_dists


def flatten(xs):
    for x in xs:
        if isinstance(x, Iterable) and not isinstance(x, (str, float)):
            yield from flatten(x)
        else:
            yield x

def mask_gaps(X, zero_idx=4):
    """ 
    X: (torch.tensor) one-hot encoded sequence data that can be reshaped into size (-1, C))
    zero_idx: (int) 0 <= zero_idx < C is the index to zero
    
    returns:
    X_new: one-hot encoded sequence where sections corresponding to zero_idx are zeroed
    """
    
    old_shape = X.size()
    X_new = X.view(-1, len(ALPHABET))
    target = torch.argmax(X_new, dim=-1)

    DEVICE = X.device

    X_new[target == zero_idx] = X_new[target == zero_idx].zero_().to(DEVICE)
    return X_new.view(old_shape)
# ------------------------------



# ACTUAL FUNCTIONS 
def make_auspice_labeling_csv():
    """
    make CSVs for valid, training, and test that can be used for visualization in auspice.us
    uses HELPER make_csv_from_df
    """

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

def get_data_dict(dset, abspath):
    """
    input
    -----
    dset: String, one of ["train", "valid", "test"]
    abspath: String, location of data directory parent dir

    output
    ------
    keys: String[], ["new_dataset", "metadata", "collection_dates", "pairs", "parents_dict", "clade_indexes", "clade_labels"]
    data_dict: Dict{}, dict with keys corresponding to above

    new_dataset - one-hot encoded sequences
    metadata - pd.DataFrame with metadata about sequences
    collection_dates - list of dates when sequences were collected (extracted from metadata)
    pairs - child/parent sequence pairs
    parents_dict - dict with keys being child, value being parent
    clade_indexes - list (of lists) where elem "i" is a list containing index of every sequence corresponding to clade "i"
    clade_labels - list where elem "i" is the clade name of sequence "i"
    """

    data_dict = dict()

    dataset = DNADataset(f"{abspath}/data/{dset}_spike.fasta")
    new_dataset = np.array([dataset[x][0].numpy() for x in range(len(dataset))])
    vals = np.array([dataset[x][1] for x in range(len(dataset))])
    # labeling
    metadata = pd.read_csv(f"{abspath}/data/all_data/all_metadata.tsv", sep="\t")
    clade_labels = [metadata.loc[metadata.name == vals[i], "clade_membership"].values[0] for i in range(len(vals))]
    dates = [metadata.loc[metadata.name == vals[i], "date"].values[0] for i in range(len(vals))]
    dates = [datetime_from_numeric(x) for x in dates]

    collection_dates = pd.DataFrame([[x] for i,x in enumerate(dates)], columns=["date"])
    collection_dates.index = pd.to_datetime(collection_dates["date"])
    collection_dates = collection_dates.groupby(pd.Grouper(freq='ME'))
    collection_dates = list(collection_dates.groups.values())
    print("collection_dates\n", collection_dates)
    collection_dates = [collection_dates[0]] + [collection_dates[i] - collection_dates[i-1] for i in range(len(collection_dates)-1, 0, -1)][::-1]
    collection_dates = list(flatten([[i for j in range(x)] for i,x in enumerate(collection_dates)]))

    clusters = np.sort(np.array(list(set(clade_labels))))
    print("\nunique clusters\n", clusters)
    get_clade = lambda x: [True if elem == x else False for elem in clade_labels]

    indexes = tuple([np.arange(len(clade_labels))[get_clade(x)] for x in clusters])

    new_vals = []
    for v in vals:
        if metadata.loc[metadata.name == v, "clade_membership"].values[0] in clusters:
            new_vals.append(v)

    # sanity check
    print("\nsanity check - len(new_vals), len(vals)\n", len(new_vals), " ", len(vals))

    parents = pd.read_csv(f"{abspath}/data/all_data/all_branches.tsv", sep="\t")
    node_dict = {x:i for i,x in enumerate(new_vals)}
    pairs = []
    for p,c in zip(parents["parent"], parents["child"]):
        i1 = node_dict.get(p, None)
        i2 = node_dict.get(c, None)

        if i1 and i2:
            pairs.append((i1,i2))

    pairs = np.array(pairs)
    get_parents_dict = np.full(new_dataset.shape[0], None)
    for (p,c) in pairs:
        get_parents_dict[c] = p


    var_names = ["new_dataset", "vals", "metadata", "clade_labels", "collection_dates", "indexes", "pairs", "get_parents_dict"]
    for k in var_names:
        data_dict[k] = eval(k)

    return var_names, data_dict

def model_eval(vae_model, new_dataset, model_type="STANDARD"):
    """
    input:
    -----
    model: either standard VAE model or bedford VAE model
    new_dataset: see get_data_dict() above

    output:
    ------
    return Z_mean, Z_logvar, recon, genome, genome_recon
    """

    assert model_type == "STANDARD" or model_type == "BEDFORD", "invalid model_type argument (should be either 'STANDARD' or 'BEDFORD')"

    X = torch.tensor(new_dataset)
    X = X.view(X.size(0), -1).to(DEVICE)
    X = mask_gaps(X,zero_idx=4)   
    
    recon = None
    Z_mean = None
    Z_logvar = None
    genome = None
    genome_recon = None

    with torch.no_grad():
        # STANDARD
        if model_type == "STANDARD":
            Z_mean, Z_logvar = vae_model.encoder.forward(X)
            recon = vae_model.decoder.forward(Z_mean)
            recon = recon.view(recon.shape[0], -1).cpu()
            Z_mean = Z_mean.cpu().numpy()
            Z_logvar = Z_logvar.cpu().numpy()

        # # BEDFORD
        elif model_type == "BEDFORD":
            recon, Z_mean, Z_logvar = vae_model.forward(X)
            recon = recon.cpu()
            Z_mean = Z_mean.cpu().numpy()
            Z_logvar = Z_logvar.cpu().numpy()
        
        print("\nRecon shape")
        print(recon.shape)

        
    converter = np.vectorize(lambda x: ALPHABET[int(x)])

    genome = np.dot(new_dataset, np.arange(len(ALPHABET)))
    genome = np.reshape(converter(genome.ravel()), genome.shape)

    genome_recon = torch.argmax(recon.view(recon.shape[0], -1, 5), dim=-1).cpu().detach().numpy()
    genome_recon = np.reshape(converter(genome_recon.ravel()), genome.shape)
    genome_recon[genome == "-"] = "-"

    return Z_mean, Z_logvar, recon.numpy(), genome, genome_recon
    

