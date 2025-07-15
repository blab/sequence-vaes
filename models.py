# code from ChatGPT
import torch
import torch.nn as nn
import numpy as np
from Bio import SeqIO
from diffusers import UNet2DModel

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

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 2 * latent_dim)  # Mean and log variance
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
            nn.Sigmoid()
        )

    def encode(self, x):
        mean_logvar = self.encoder(x)
        mean, logvar = mean_logvar[:, :LATENT_DIM], mean_logvar[:, LATENT_DIM:]
        return mean, logvar

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        return self.decode(z), mean, logvar

class DiffusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = UNet2DModel(
            sample_size=WIDTH,                          # Width of the latent representation
            in_channels=len(ALPHABET),                  # Input channel for latent vectors
            out_channels=len(ALPHABET),                 # Output channel
            layers_per_block=2,                         # Number of layers per block
            block_out_channels=(64, 128),              # Match the number of down_block_types
            down_block_types=("DownBlock2D", "AttnDownBlock2D"),
            up_block_types=("UpBlock2D", "AttnUpBlock2D")
        )

    def forward(self, x, timesteps):
        return self.model(x, timesteps)
