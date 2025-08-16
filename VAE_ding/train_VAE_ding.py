import numpy as np
import matplotlib.pyplot as plt
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from sys import exit
import sys

sys.path.append("../../PEVAE_Paper/simulated_msa/script")
from VAE_model import *

## read multiple sequence alignment in binary representation
with open("./output/training_msa_leaf_binary.pkl", 'rb') as file_handle:
    msa_binary = pickle.load(file_handle)    
num_seq = msa_binary.shape[0]
len_protein = msa_binary.shape[1]
num_res_type = msa_binary.shape[2]
msa_binary = msa_binary.reshape((num_seq, -1))
msa_binary = msa_binary.astype(np.float32)

## each sequence has a label
with open("./output/training_msa_leaf_keys.pkl", 'rb') as file_handle:
    msa_keys = pickle.load(file_handle)    

## sequences in msa are weighted. Here sequences are assigned
## the same weights
msa_weight = np.ones(num_seq) / num_seq
msa_weight = msa_weight.astype(np.float32)


# VALIDATION
batch_size = num_seq
train_data = MSA_Dataset(msa_binary, msa_weight, msa_keys)
train_data_loader = DataLoader(train_data, batch_size = batch_size, shuffle = True)

eval_type = "valid"
with open(f"./output/{eval_type}_msa_leaf_binary.pkl", 'rb') as file_handle:
    valid_msa_binary = pickle.load(file_handle)    
valid_num_seq = valid_msa_binary.shape[0]
valid_len_protein = valid_msa_binary.shape[1]
valid_num_res_type = valid_msa_binary.shape[2]
valid_msa_binary = valid_msa_binary.reshape((valid_num_seq, -1))
valid_msa_binary = valid_msa_binary.astype(np.float32)

## each sequence has a label
with open(f"./output/{eval_type}_msa_leaf_keys.pkl", 'rb') as file_handle:
    valid_msa_keys = pickle.load(file_handle)

## sequences in msa are weighted. Here sequences are assigned
## the same weights
valid_msa_weight = np.ones(valid_num_seq) / valid_num_seq
valid_msa_weight = valid_msa_weight.astype(np.float32)

valid_batch_size = valid_num_seq
valid_train_data = MSA_Dataset(valid_msa_binary, valid_msa_weight, valid_msa_keys)
valid_train_data_loader = DataLoader(valid_train_data, batch_size = valid_batch_size, shuffle = True)

print(valid_len_protein)
print(valid_num_res_type)
print(len_protein)
print(num_res_type)


## here we use only one hidden layer with 100 neurons. If you want to use more
## hidden layers, change the parameter num_hidden_units. For instance, changing
## it to [100, 150] will use two hidden layers with 100 and 150 neurons.
vae = VAE(num_aa_type = 21,
          dim_latent_vars = 2,
          dim_msa_vars = len_protein*num_res_type,
          num_hidden_units = [100])
vae.cuda()

weight_decay = 0.01
optimizer = optim.Adam(vae.parameters(), weight_decay = 0.01)
num_epoches = 10000
train_loss_epoch = []
# test_loss_epoch = []
valid_loss_epoch = []

print("training seq:")
print(next(iter(train_data_loader))[0].shape)
print("valid seq:")
print(next(iter(valid_train_data_loader))[0].shape)

for epoch in range(num_epoches):
    running_loss = []    
    for idx, data in enumerate(train_data_loader):
        msa, weight, _ = data
        msa = msa.cuda()
        weight = weight.cuda()
        loss = (-1)*vae.compute_weighted_elbo(msa, weight)
        optimizer.zero_grad()
        loss.backward()        
        optimizer.step()
        
        print("Epoch: {:>4}, Step: {:>4}, loss: {:>4.2f}".format(epoch, idx, loss.data.item()), flush = True)
        running_loss.append(loss.data.item())        
    train_loss_epoch.append(np.mean(running_loss))

    running_loss = [] 
    with torch.no_grad():
        for i, data in enumerate(valid_train_data_loader):
            msa, weight, _ = data
            msa = msa.cuda()
            weight = weight.cuda()
            loss = (-1)*vae.compute_weighted_elbo(msa, weight)
            running_loss.append(loss.data.item())  
        
        valid_loss_epoch.append(np.mean(running_loss))

    if epoch % 1000 == 0 and epoch > 0:
        torch.save(vae.state_dict(), f"./output/vae_{epoch}.model")

torch.save(vae.state_dict(), f"./output/vae_FINAL.model")

with open('./output/loss.pkl', 'wb') as file_handle:
    pickle.dump({'train_loss_epoch': train_loss_epoch, 'valid_loss_epoch': valid_loss_epoch}, file_handle)

fig = plt.figure(0)
fig.clf()
plt.plot(train_loss_epoch, label = "train", color = 'r')
plt.plot(valid_loss_epoch, label = "valid", color = 'b')
#plt.ylim((140, 180))
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.title("Loss")
fig.savefig("./output/loss.png")
#plt.show()
