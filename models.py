# code from ChatGPT
import torch
import torch.nn as nn
import numpy as np
from Bio import SeqIO

# Constants
LATENT_DIM = 80    # Dimensionality of latent space len(ALPHABET) * HEIGHT * WIDTH = 5 * 4 * 4 = 80
HEIGHT = 4          # Height of the latent tensor
WIDTH = 4           # Width of the latent tensor
ALPHABET = "ATGC-"  # DNA alphabet, with "-" as gap/unknown character
SEQ_LENGTH = 3822   # Length of DNA sequence to generate (for spike gene)

class DNADataset(torch.utils.data.Dataset):
    def __init__(self, fasta_file):
        self.records = list(SeqIO.parse(fasta_file, "fasta"))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        sequence = str(record.seq)
        encoded = self.one_hot_encode(sequence)
        return torch.tensor(encoded, dtype=torch.float32), record.id

    @staticmethod
    def one_hot_encode(sequence):
        mapping = {char: i for i, char in enumerate(ALPHABET)}
        encoded = np.zeros((len(sequence), len(ALPHABET)), dtype=np.float32)
        for i, nucleotide in enumerate(sequence):
            encoded[i, mapping.get(nucleotide, mapping["-"])] = 1.0
        return encoded

class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim, non_linear_activation):
        super().__init__()
        self.input_channels = len(ALPHABET)
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.non_linear_activation = non_linear_activation

        self.seq_len = input_dim

        self.encode = nn.Sequential(
            nn.Linear(self.seq_len, 512),
            self.non_linear_activation,
            nn.Linear(512, 256),
            self.non_linear_activation
        )
        self.fc_mean = nn.Linear(256, self.latent_dim)
        self.fc_logvar = nn.Linear(256, self.latent_dim)

    def forward(self, x):
        x = self.encode(x)
        mean = self.fc_mean(x)
        log_var = self.fc_logvar(x)
        return mean, log_var

class Decoder(nn.Module):
    def __init__(self, input_dim, latent_dim, non_linear_activation):
        super().__init__()
        self.input_channels = len(ALPHABET)
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.non_linear_activation = non_linear_activation
        # self.last_non_linear_activation = nn.Sigmoid()

        self.means = nn.ModuleList([
            nn.Linear(self.latent_dim, 256),
            nn.Linear(256,512),
            nn.Linear(512, self.input_dim * self.input_channels) # last layer
        ])

        for mu in self.means:
            nn.init.constant_(mu.bias,0.1)

    def forward(self, x):
        for i in range(2):
            x = self.means[i](x)
            x = self.non_linear_activation(x)

        x = self.means[-1](x)
        # x = self.last_non_linear_activation(x)
        x = x.view(-1,self.input_dim, self.input_channels)
        # x = F.softmax(x, dim=-1)
        x = F.log_softmax(x, dim=-1)
        x = x.view(-1,self.input_dim * self.input_channels)
        return x

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.non_linear_activation = nn.ReLU()

        self.encoder = Encoder(input_dim, latent_dim, self.non_linear_activation)
        self.decoder = Decoder(input_dim // len(ALPHABET), latent_dim, self.non_linear_activation)

    def train_vae(self, dataloader, epochs, optimizer):
        mse_loss = nn.MSELoss()
        for epoch in range(epochs):
            self.train()
            epoch_loss = 0.0
            epoch_recon_loss = 0.0
            epoch_kl_loss = 0.0
            for i, record_set in enumerate(dataloader):
                batch, _ = record_set  # Unpack sequence tensor and record_id
                # batch = batch.to(DEVICE)
                batch = batch.view(batch.size(0), -1).to(DEVICE)  # Flatten one-hot sequences

                # Train VAE
                mean, logvar = self.encoder.forward(batch)

                eps = torch.randn_like(mean)
                std = torch.exp(0.5 * logvar)
                z = torch.mul(eps, std) + mean

                recon = self.decoder.forward(z)

                recon_loss = F.binary_cross_entropy_with_logits(recon, batch, reduction='sum') / batch.shape[0]
                kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp()) / batch.shape[0]
                beta = 0.1  # tunable parameter
                vae_loss = recon_loss + beta * kl_loss

                optimizer.zero_grad()
                vae_loss.backward()
                optimizer.step()

                epoch_loss += vae_loss.item()
                epoch_recon_loss += recon_loss.item()
                epoch_kl_loss += kl_loss.item()


            print(f"Epoch {epoch+1}/{epochs} - VAE Loss: {epoch_loss / len(dataloader):.4f} - Reconstruction Loss: {epoch_recon_loss / len(dataloader):.4f} - KL Loss: {epoch_kl_loss / len(dataloader):.4f}")
