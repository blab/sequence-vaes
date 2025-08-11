# code from ChatGPT
import argparse
import os
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from models import VAE, DNADataset, ALPHABET, SEQ_LENGTH, LATENT_DIM
import torch.nn as nn

BATCH_SIZE = 64
EPOCHS = 50

def main(args):
    # DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DEVICE = args.device

    # training data
    train_dataset = DNADataset(args.input_alignment)
    print("training data number: %d"%(len(train_dataset)))
    dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

    # validation data
    valid_dataloader = None
    if args.input_valid_alignment:
        valid_dataset = DNADataset(args.input_valid_alignment)
        print("validation data number: %d"%(len(valid_dataset)))
        valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

    input_dim = 29903 * len(ALPHABET)
    vae_model = VAE(input_dim=input_dim, latent_dim=50).to(DEVICE)

    if args.input_vae_model and os.path.exists(args.input_vae_model):
        print(f"Loading VAE model from {args.input_vae_model}")
        vae_model.load_state_dict(torch.load(args.input_vae_model, map_location=DEVICE))
    else:
        print(f"Initializing new VAE model")

    vae_params = sum(p.numel() for p in vae_model.parameters() if p.requires_grad)
    print(f"VAE parameters: {vae_params}")

    optimizer = Adam(vae_model.parameters(), lr=1e-3)
    vae_model.train_vae(dataloader, 
                        EPOCHS,     
                        optimizer, 
                        valid_dataloader=valid_dataloader, 
                        train_logs=args.train_logs, 
                        valid_logs=args.valid_logs, 
                        model_save_path=args.output_vae_model, 
                        DEVICE=DEVICE)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a VAE model on DNA sequences.")
    parser.add_argument("--input-alignment", type=str, help="file Path to the training input aligned FASTA file.")
    parser.add_argument("--train-logs", type=str, default=None, help="filepath to store training results")
    parser.add_argument("--output-vae-model", type=str, default=None, help="dir Path to save the trained VAE model.")
    parser.add_argument("--device", type=str, default=None, help="cuda or cpu")

    parser.add_argument("--input-vae-model", type=str, default=None, help="file Path to the pretrained VAE model.")
    parser.add_argument("--input-valid-alignment", type=str, default=None, help="file Path to the validation input aligned FASTA file.")
    parser.add_argument("--valid-logs", type=str, default=None, help="filepath to store validation results")

    args = parser.parse_args(["--input-alignment", "../data/training/training_aligned.fasta",
                            "--train-logs", "./results/train_logs.json",
                            "--output-vae-model", "./model_saves",
                            "--device", "cuda:1",
                            "--input-valid-alignment", "../data/valid/valid_aligned.fasta",
                            "--valid-logs", "./results/valid_logs.json"])

    print("args:\n------------------------------")
    for k,v in sorted(vars(args).items()):
        print(f"%-40s:\t{v}"%f"{k}")
    print()

    main(args)
