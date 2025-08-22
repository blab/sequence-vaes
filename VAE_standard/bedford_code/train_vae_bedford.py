import argparse
import os

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import json

from models_bedford import (
    VAE,
    DNADataset,
    ALPHABET,
    SEQ_LENGTH,
    LATENT_DIM,
)

# # ----------------------------------------------------------------------------
# # Device
# # ----------------------------------------------------------------------------
# if torch.cuda.is_available():
#     DEVICE = torch.device("cuda")
# elif torch.backends.mps.is_available():
#     DEVICE = torch.device("mps")
# else:
#     DEVICE = torch.device("cpu")

# ----------------------------------------------------------------------------
# Hyper-parameters
# ----------------------------------------------------------------------------
BATCH_SIZE = 512
EPOCHS = 80
LR = 1e-3

# — KL regularisation —
BETA_MAX = 0.05          # target β when warm-up completes
WARMUP_STEPS = 5000      # gradient steps to reach BETA_MAX
FREE_BITS = 1.0          # minimum nats per latent dim

# ----------------------------------------------------------------------------
# Training loop
# ----------------------------------------------------------------------------

def train_vae(model: VAE, loader: DataLoader, epochs: int, optimizer: Adam, loader_valid: DataLoader = None, args=None):
    """Train a VAE using CE reconstruction, β-annealing, and free bits."""

    DEVICE = args.device
    ce_loss = nn.CrossEntropyLoss(reduction="mean")
    vocab_size = len(ALPHABET)
    global_step = 0

    train_losses = []
    valid_losses = []
    best_v_loss = float('inf')
    best_model = model.state_dict()
    model_save_path = args.output_vae_model
    train_logs = args.train_logs
    valid_logs = args.valid_logs

    for epoch in range(epochs):
        model.train()
        epoch_loss = epoch_recon = epoch_kl = 0.0
        valid_epoch_loss = valid_epoch_recon = valid_epoch_kl = 0.0

        for batch, _ in loader:  # batch: (B, L, V)
            global_step += 1
            beta = BETA_MAX * min(1.0, global_step / WARMUP_STEPS)

            # --------------------------------------------------------------
            # Prepare data
            # --------------------------------------------------------------
            batch = batch.to(DEVICE)
            flat_batch = batch.view(batch.size(0), -1)  # (B, L*V)

            # --------------------------------------------------------------
            # Forward pass
            # --------------------------------------------------------------
            logits, mu, logvar = model(flat_batch)

            # --------------------------------------------------------------
            # Reconstruction loss (categorical cross-entropy)
            # --------------------------------------------------------------
            logits = logits.view(-1, vocab_size)        # (B*L, V)
            targets = batch.argmax(dim=-1).view(-1)     # (B*L,)
            recon_loss = ce_loss(logits, targets)

            # --------------------------------------------------------------
            # KL divergence with free bits
            # --------------------------------------------------------------
            kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())  # (B, latent)
            kl_per_dim = torch.clamp(kl_per_dim, min=FREE_BITS)          # apply free bits
            kl = kl_per_dim.sum(dim=1).mean()                            # average over batch

            # --------------------------------------------------------------
            # Total loss and optimisation
            # --------------------------------------------------------------
            loss = recon_loss + beta * kl
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Logging accumulators (already averaged by CE)
            epoch_loss += loss.item()
            epoch_recon += recon_loss.item()
            epoch_kl += kl.item()

        n_batches = len(loader)
        print(
            f"Epoch {epoch + 1}/{epochs} - "
            f"β: {beta:.4f} - "
            f"VAE Loss: {epoch_loss / n_batches:.4f} - "
            f"Recon.: {epoch_recon / n_batches:.4f} - "
            f"KL: {epoch_kl / n_batches:.4f}"
        )
    
        with torch.no_grad():
            for batch, _ in loader_valid:  # batch: (B, L, V)
                beta = BETA_MAX * min(1.0, global_step / WARMUP_STEPS)

                batch = batch.to(DEVICE)
                flat_batch = batch.view(batch.size(0), -1)  # (B, L*V)

                logits, mu, logvar = model(flat_batch)

                logits = logits.view(-1, vocab_size)        # (B*L, V)
                targets = batch.argmax(dim=-1).view(-1)     # (B*L,)
                recon_loss = ce_loss(logits, targets)

                kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())  # (B, latent)
                kl_per_dim = torch.clamp(kl_per_dim, min=FREE_BITS)          # apply free bits
                kl = kl_per_dim.sum(dim=1).mean()                            # average over batch

                loss = recon_loss + beta * kl

                valid_epoch_loss += loss.item()
                valid_epoch_recon += recon_loss.item()
                valid_epoch_kl += kl.item()
            
            if valid_epoch_loss < best_v_loss:
                best_model = model.state_dict()
            
        n_batches = len(loader)
        train_losses.append((epoch_loss / n_batches,
                            epoch_recon / n_batches,
                            epoch_kl / n_batches))
        
        n_batches = len(loader_valid)
        valid_losses.append((valid_epoch_loss / n_batches, 
                            valid_epoch_recon / n_batches, 
                            valid_epoch_kl / n_batches))

        with open(train_logs, "w") as f:
            json.dump(train_losses, f)
        with open(valid_logs, "w") as f:
            json.dump(valid_losses, f)
    
        n_batches = len(loader_valid)
        print(
            "VALID"
            f"Epoch {epoch + 1}/{epochs} - "
            f"β: {beta:.4f} - "
            f"VAE Loss: {valid_epoch_loss / n_batches:.4f} - "
            f"Recon.: {valid_epoch_recon / n_batches:.4f} - "
            f"KL: {valid_epoch_kl / n_batches:.4f}\n"
        )

    torch.save(best_model, f"{model_save_path}/BEST_vae_ce_anneal.pth")


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def main(args):
    DEVICE = args.device
    dataset = DNADataset(args.input_alignment)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    dataset_valid = DNADataset(args.input_valid_alignment)
    loader_valid = DataLoader(dataset_valid, batch_size=len(dataset_valid), shuffle=True, pin_memory=True)  # just do one pass

    model = VAE(input_dim=len(ALPHABET) * SEQ_LENGTH, latent_dim=LATENT_DIM).to(DEVICE)

    if args.input_vae_model and os.path.exists(args.input_vae_model):
        print(f"Loading pretrained weights from {args.input_vae_model}")
        model.load_state_dict(torch.load(args.input_vae_model, map_location=DEVICE))
    else:
        print("Initialising new VAE model")

    print("Trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    optimizer = Adam(model.parameters(), lr=LR)
    train_vae(model, loader, EPOCHS, optimizer, loader_valid=loader_valid, args=args)

    torch.save(model.state_dict(), f"{args.output_vae_model}/FINAL_vae_ce_anneal.pth")
    print(f"Final model saved to {args.output_vae_model}/FINAL_vae_ce_anneal.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train VAE with CE + KL annealing + free bits")
    parser.add_argument("--input-alignment", required=True, help="Path to FASTA alignment")
    parser.add_argument("--input-vae-model", help="Optional checkpoint to resume from")
    parser.add_argument("--output-vae-model", default="vae_ce_anneal.pth", help="Checkpoint destination")

    parser.add_argument("--input-valid-alignment")
    parser.add_argument("--train-logs")
    parser.add_argument("--valid-logs")
    parser.add_argument("--device")
    # args = parser.parse_args()
    
    args = parser.parse_args(["--input-alignment", "../../data/training_spike.fasta",
                            "--input-valid-alignment", "../../data/valid_spike.fasta",
                            # "--output-vae-model", "./results_bedford/vae_ce_anneal.pth",
                            "--output-vae-model", "./results_bedford",
                            "--train-logs", "./results_bedford/train_logs.json",
                            "--device", "cuda",
                            "--valid-logs", "./results_bedford/valid_logs.json"])

    print("args:\n------------------------------")
    for k,v in sorted(vars(args).items()):
        print(f"%-40s:\t{v}"%f"{k}")
    print()

    main(args)
