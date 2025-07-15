# code from ChatGPT
import argparse
import os
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from models import VAE, DiffusionModel, DNADataset, ALPHABET, LATENT_DIM, HEIGHT, WIDTH
from diffusers import DDPMScheduler
import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
EPOCHS = 10

def train_diffusion(vae_model, diffusion_model, scheduler, dataloader, epochs, optimizer):
    mse_loss = nn.MSELoss()
    for epoch in range(epochs):
        diffusion_model.train()
        epoch_loss = 0.0
        for record_set in dataloader:
            batch, _ = record_set  # Unpack sequence tensor and record_id
            batch = batch.view(batch.size(0), -1).to(DEVICE)  # Flatten one-hot sequences

            # Compute latent embeddings dynamically using the VAE
            with torch.no_grad():
                latent, _ = vae_model.encode(batch)
            latent = latent.view(latent.size(0), len(ALPHABET), HEIGHT, WIDTH)  # Reshape latent representations

            noise = torch.randn_like(latent).to(DEVICE)
            timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (latent.size(0),)).to(DEVICE)
            noisy_latent = scheduler.add_noise(latent, noise, timesteps)
            predicted_noise = diffusion_model(noisy_latent, timesteps).sample  # Access tensor from UNet2DOutput

            loss = mse_loss(predicted_noise, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs} - Diffusion Loss: {epoch_loss / len(dataloader):.4f}")


def main(args):
    # Load dataset
    dataset = DNADataset(args.input_alignment)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

    # Load pre-trained VAE model
    input_dim = len(ALPHABET) * LATENT_DIM
    vae_model = VAE(input_dim=input_dim, latent_dim=LATENT_DIM).to(DEVICE)
    print(f"Loading VAE model from {args.input_vae_model}")
    vae_model.load_state_dict(torch.load(args.input_vae_model, map_location=DEVICE, weights_only=False))
    vae_model.eval()  # Ensure VAE is in evaluation mode

    # Initialize or load Diffusion model
    diffusion_model = DiffusionModel().to(DEVICE)
    if args.input_diffusion_model and os.path.exists(args.input_diffusion_model):
        print(f"Loading Diffusion model from {args.input_diffusion_model}")
        diffusion_model.load_state_dict(torch.load(args.input_diffusion_model, map_location=DEVICE, weights_only=False))
    else:
        print(f"Initializing new diffusion model")

    diffusion_params = sum(p.numel() for p in diffusion_model.parameters() if p.requires_grad)
    print(f"Diffusion model parameters: {diffusion_params}")

    scheduler = DDPMScheduler(num_train_timesteps=1000)
    optimizer = Adam(diffusion_model.parameters(), lr=1e-4)

    train_diffusion(vae_model, diffusion_model, scheduler, dataloader, EPOCHS, optimizer)

    # Save the trained diffusion model
    torch.save(diffusion_model.state_dict(), args.output_diffusion_model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Diffusion Model using latent embeddings dynamically computed by a pre-trained VAE.")
    parser.add_argument("--input-alignment", type=str, default="data/alignment.fasta", help="Path to the input FASTA file.")
    parser.add_argument("--input-vae-model", type=str, default="models/vae.pth", help="Path to the pre-trained VAE model.")
    parser.add_argument("--input-diffusion-model", type=str, default=None, help="Path to the pretrained Diffusion model.")
    parser.add_argument("--output-diffusion-model", type=str, default="models/diffusion.pth", help="Path to save the trained Diffusion model.")

    args = parser.parse_args()

    print(f"Using device: {DEVICE}")

    main(args)
