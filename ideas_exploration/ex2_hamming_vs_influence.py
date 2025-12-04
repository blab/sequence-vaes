from transformers import AutoTokenizer, AutoModelForMaskedLM, PreTrainedTokenizerFast, DataCollatorForLanguageModeling
import torch
import torch.nn as nn
import sys
import numpy as np
sys.path.append("../VAE_standard")
from models import DNADataset, ALPHABET, SEQ_LENGTH, LATENT_DIM, VAE

from matplotlib import pyplot as plt

sys.path.append("..")
import utils

from Bio import SeqIO
from Bio.Seq import Seq

from devinterp.utils import (
    EvaluateFn,
    EvalResults,
)

from BIF_sampler import (
    BIFEstimator,
    estimate_bif
)

import pandas as pd

MAX_TOKEN_LENGTH = 510
BATCH_SIZE=80
BIF_BATCH_SIZE=80
num_masks = 3

TEST_SEQ = 1
# TRAIN_CUTOFF = 3000
# TEST_TOKEN = 0

DEVICE = "cuda"

names = []
sequences = []

def get_ex2_data(tokenizer, file_path="../data/training_spike.fasta"):
    for i, record in enumerate(SeqIO.parse(file_path, "fasta")):
        names.append(record.name)
        seq = record.seq
        seq = seq.replace("-","")
        n = len(seq)
        sequences.append(seq[:(n - (n%3))])
    print("done extracting sequences!")

    AA_seqs = [str(Seq(x).translate())[:-1] for x in sequences]
    aa_drop_na = []
    names_drop_na = []
    for n,s in zip(names,AA_seqs):
        # only choose AA sequences WITHOUT stop codon in the middle
        if "*" not in s:
            aa_drop_na.append(s)
            names_drop_na.append(n)
    print("done extracting AAs!")

    unique_aa_seqs = list(np.unique(aa_drop_na))
    train_aa_seqs = [x[i*(MAX_TOKEN_LENGTH // 2):(i+2)*(MAX_TOKEN_LENGTH // 2)] for x in unique_aa_seqs for i in range(4)]
    unique_aa_seqs = [x[:MAX_TOKEN_LENGTH] for x in unique_aa_seqs]
    
    train_data = tokenizer(text=train_aa_seqs, return_tensors="pt", add_special_tokens=False, truncation=False, padding=True, padding_side="right")["input_ids"]
    bif_data = tokenizer(text=unique_aa_seqs, return_tensors="pt", add_special_tokens=False, truncation=False, padding=True, padding_side="right")["input_ids"]

    return {"aa_drop_na":aa_drop_na, 
            "names_drop_na":names_drop_na, 
            "unique_aa_seqs":unique_aa_seqs, 
            "train_data":train_data,
            "bif_data":bif_data}
