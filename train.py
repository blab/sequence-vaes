# code from ChatGPT
import argparse
import os
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from models import VAE, DiffusionModel, DNADataset, ALPHABET, SEQ_LENGTH, LATENT_DIM, HEIGHT, WIDTH
from diffusers import DDPMScheduler
import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
EPOCHS = 10

def train(vae_model, diffusion_model, scheduler, dataloader, epochs, vae_optimizer, diffusion_optimizer):
    mse_loss = nn.MSELoss()
    for epoch in range(epochs):
        vae_model.train()
        diffusion_model.train()
        for record_set in dataloader:
            batch, _ = record_set  # Unpack sequence tensor and record_id
            batch = batch.view(batch.size(0), -1).to(DEVICE)  # Flatten one-hot sequences

            # Train VAE
            recon, mean, logvar = vae_model(batch)
            recon_loss = mse_loss(recon, batch)
            kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
            beta = 0.1  # tunable parameter
            vae_loss = recon_loss + beta * kl_loss
            vae_optimizer.zero_grad()
            vae_loss.backward()
            vae_optimizer.step()

            # Train Diffusion Model
            latent, _ = vae_model.encode(batch)
            latent = latent.view(latent.size(0), len(ALPHABET), HEIGHT, WIDTH)  # Reshape to [batch, channels, height, width]
            noise = torch.randn_like(latent).to(DEVICE)
            timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (latent.size(0),)).to(DEVICE)
            noisy_latent = scheduler.add_noise(latent, noise, timesteps)
            predicted_noise = diffusion_model(noisy_latent, timesteps).sample  # Access the tensor from UNet2DOutput
            diffusion_loss = mse_loss(predicted_noise, noise)

            diffusion_optimizer.zero_grad()
            diffusion_loss.backward()
            diffusion_optimizer.step()

        print(f"Epoch {epoch+1}/{epochs} - VAE Loss: {vae_loss.item():.4f} - Diffusion Loss: {diffusion_loss.item():.4f}")

def main(args):
    dataset = DNADataset(args.input_alignment)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

    input_dim = len(ALPHABET) * SEQ_LENGTH
    vae_model = VAE(input_dim=input_dim, latent_dim=LATENT_DIM).to(DEVICE)
    if args.input_vae_model and os.path.exists(args.input_vae_model):
        print(f"Loading VAE model from {args.input_vae_model}")
        vae_model.load_state_dict(torch.load(args.input_vae_model, map_location=DEVICE, weights_only=False))
    else:
        print(f"Initializing new VAE model")

    vae_params = sum(p.numel() for p in vae_model.parameters() if p.requires_grad)
    print(f"VAE parameters: {vae_params}")

    diffusion_model = DiffusionModel().to(DEVICE)
    if args.input_diffusion_model and os.path.exists(args.input_diffusion_model):
        print(f"Loading Diffusion model from {args.input_diffusion_model}")
        diffusion_model.load_state_dict(torch.load(args.input_diffusion_model, map_location=DEVICE, weights_only=False))
    else:
        print(f"Initializing new diffusion model")

    diffusion_params = sum(p.numel() for p in diffusion_model.parameters() if p.requires_grad)
    print(f"Diffusion model parameters: {diffusion_params}")

    scheduler = DDPMScheduler(num_train_timesteps=1000)

    vae_optimizer = Adam(vae_model.parameters(), lr=1e-3)
    diffusion_optimizer = Adam(diffusion_model.parameters(), lr=1e-4)

    train(vae_model, diffusion_model, scheduler, dataloader, EPOCHS, vae_optimizer, diffusion_optimizer)

    # Save models
    torch.save(vae_model.state_dict(), args.output_vae_model)
    torch.save(diffusion_model.state_dict(), args.output_diffusion_model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Latent Diffusion Model on DNA sequences.")
    parser.add_argument("--input-alignment", type=str, default="data/alignment.fasta", help="Path to the input FASTA file.")
    parser.add_argument("--input-vae-model", type=str, default=None, help="Path to the pretrained VAE model.")
    parser.add_argument("--input-diffusion-model", type=str, default=None, help="Path to the pretrained Diffusion model.")
    parser.add_argument("--output-vae-model", type=str, default="models/vae.pth", help="Path to save the trained VAE model.")
    parser.add_argument("--output-diffusion-model", type=str, default="models/diffusion.pth", help="Path to save the trained Diffusion model.")

    args = parser.parse_args()

    print(f"Using device: {DEVICE}")

    main(args)
