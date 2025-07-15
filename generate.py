# code from ChatGPT
import argparse
import os
import torch
from models import VAE, DiffusionModel, ALPHABET, SEQ_LENGTH, HEIGHT, WIDTH, LATENT_DIM
from diffusers import DDPMScheduler

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_sequences(vae_model_path, diffusion_model_path, count, batch_size):

    # Load trained models
    input_dim = len(ALPHABET) * SEQ_LENGTH
    vae_model = VAE(input_dim=input_dim, latent_dim=LATENT_DIM).to(DEVICE)
    print(f"Loading VAE model from {vae_model_path}")
    vae_model.load_state_dict(torch.load(vae_model_path, map_location=DEVICE, weights_only=False))
    vae_model.eval()

    diffusion_model = DiffusionModel().to(DEVICE)
    print(f"Loading Diffusion model from {diffusion_model_path}")
    diffusion_model.load_state_dict(torch.load(diffusion_model_path, map_location=DEVICE, weights_only=False))
    diffusion_model.eval()

    scheduler = DDPMScheduler(num_train_timesteps=1000)

    generated_sequences = []
    print(f"Generating sequences in batches of {batch_size}")

    num_batches = (count + batch_size - 1) // batch_size  # Ceiling division to cover all sequences
    for batch_idx in range(num_batches):
        print(f"Generating batch {batch_idx + 1}/{num_batches}")
        current_batch_size = min(batch_size, count - len(generated_sequences))

        # Generate noise for the batch
        latent = torch.randn((current_batch_size, len(ALPHABET), HEIGHT, WIDTH), device=DEVICE)

        # Reverse diffusion process
        for t in reversed(range(scheduler.config.num_train_timesteps)):
            latent = scheduler.step(diffusion_model(latent, torch.tensor([t] * current_batch_size, device=DEVICE)).sample, t, latent).prev_sample

        # Decode latent space to one-hot sequences
        latent_flat = latent.view(current_batch_size, -1)
        reconstructed_sequences = vae_model.decode(latent_flat).view(current_batch_size, -1, len(ALPHABET))

        for seq in reconstructed_sequences:
            decoded_sequence = "".join(ALPHABET[torch.argmax(x).item()] for x in seq)
            generated_sequences.append(decoded_sequence)

    return generated_sequences

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate DNA sequences from trained models")
    parser.add_argument("--input-vae-model", type=str, default="models/vae.pth", help="Path to the trained VAE model")
    parser.add_argument("--input-diffusion-model", type=str, default="models/diffusion.pth", help="Path to the trained Diffusion model")
    parser.add_argument("--output-alignment", type=str, default="results/generated.fasta", help="FASTA file to output sequences to")
    parser.add_argument("--count", type=int, default=10, help="Number of DNA sequences to generate")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for generation")
    args = parser.parse_args()

    print(f"Using device: {DEVICE}")

    sequences = generate_sequences(args.input_vae_model, args.input_diffusion_model, args.count, args.batch_size)

    if not os.path.exists("results/"):
        os.makedirs("results/")

    with open(args.output_alignment, 'w') as f:
        for i, seq in enumerate(sequences):
            print(f">seq_{i + 1}", file=f)
            print(seq, file=f)
