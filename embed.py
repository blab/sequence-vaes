# code from ChatGPT
import argparse
import os
import torch
from models import VAE, DNADataset, ALPHABET, SEQ_LENGTH, LATENT_DIM
import pandas as pd

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_embeddings(fasta_file, vae_model_path, output_file):

    # Load the VAE model
    input_dim = len(ALPHABET) * SEQ_LENGTH
    vae_model = VAE(input_dim=input_dim, latent_dim=LATENT_DIM).to(DEVICE)
    print(f"Loading VAE model from {vae_model_path}")
    vae_model.load_state_dict(torch.load(vae_model_path, map_location=DEVICE, weights_only=False))
    vae_model.eval()

    # Load the dataset
    dataset = DNADataset(fasta_file)
    embeddings = []
    sequence_ids = []

    # Process each sequence
    with torch.no_grad():
        for sequence_tensor, record_id in dataset:
            sequence_tensor = sequence_tensor.view(1, -1).to(DEVICE)  # Flatten and add batch dimension
            mean, _ = vae_model.encode(sequence_tensor)  # Get the mean embedding
            embeddings.append(mean.cpu().numpy().flatten())
            sequence_ids.append(record_id)  # Use the original strain name as the row name

    # Save embeddings to a TSV file
    embeddings_df = pd.DataFrame(embeddings, index=sequence_ids)
    embeddings_df.to_csv(output_file, sep="\t", header=False, index=True)
    print(f"Embeddings saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate latent space embeddings for DNA sequences")
    parser.add_argument("--input-alignment", type=str, default="data/alignment.fasta", help="Path to the input FASTA file")
    parser.add_argument("--input-vae-model", type=str, default="models/vae.pth", help="Path to the trained VAE model")
    parser.add_argument("--output-embeddings", type=str, default="results/embeddings.tsv", help="Path to save the output embeddings (TSV format)")
    args = parser.parse_args()

    print(f"Using device: {DEVICE}")

    if not os.path.exists("results/"):
        os.makedirs("results/")

    generate_embeddings(args.input_alignment, args.input_vae_model, args.output_embeddings)
