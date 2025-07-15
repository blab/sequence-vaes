# code from ChatGPT
import argparse
import os
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from models import VAE, DNADataset, ALPHABET, SEQ_LENGTH, LATENT_DIM
import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
EPOCHS = 10

def train_vae(vae_model, dataloader, epochs, optimizer):
    mse_loss = nn.MSELoss()
    for epoch in range(epochs):
        vae_model.train()
        epoch_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_kl_loss = 0.0
        for record_set in dataloader:
            batch, _ = record_set  # Unpack sequence tensor and record_id
            batch = batch.view(batch.size(0), -1).to(DEVICE)  # Flatten one-hot sequences

            # Train VAE
            recon, mean, logvar = vae_model(batch)
            recon_loss = mse_loss(recon, batch)
            kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
            beta = 0.1  # tunable parameter
            vae_loss = recon_loss + beta * kl_loss

            optimizer.zero_grad()
            vae_loss.backward()
            optimizer.step()

            epoch_loss += vae_loss.item()
            epoch_recon_loss += recon_loss.item()
            epoch_kl_loss += kl_loss.item()

        print(f"Epoch {epoch+1}/{epochs} - VAE Loss: {epoch_loss / len(dataloader):.4f} - Reconstruction Loss: {epoch_recon_loss / len(dataloader):.4f} - KL Loss: {epoch_kl_loss / len(dataloader):.4f}")

def main(args):
    dataset = DNADataset(args.input_alignment)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

    input_dim = len(ALPHABET) * SEQ_LENGTH
    vae_model = VAE(input_dim=input_dim, latent_dim=LATENT_DIM).to(DEVICE)
    if args.input_vae_model and os.path.exists(args.input_vae_model):
        print(f"Loading VAE model from {args.input_vae_model}")
        vae_model.load_state_dict(torch.load(args.input_vae_model, map_location=DEVICE))
    else:
        print(f"Initializing new VAE model")

    vae_params = sum(p.numel() for p in vae_model.parameters() if p.requires_grad)
    print(f"VAE parameters: {vae_params}")

    optimizer = Adam(vae_model.parameters(), lr=1e-3)
    train_vae(vae_model, dataloader, EPOCHS, optimizer)

    # Save the trained VAE model
    torch.save(vae_model.state_dict(), args.output_vae_model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a VAE model on DNA sequences.")
    parser.add_argument("--input-alignment", type=str, required=True, help="Path to the input FASTA file.")
    parser.add_argument("--input-vae-model", type=str, default=None, help="Path to the pretrained VAE model.")
    parser.add_argument("--output-vae-model", type=str, default="models/vae.pth", help="Path to save the trained VAE model.")

    args = parser.parse_args()

    print(f"Using device: {DEVICE}")

    main(args)
