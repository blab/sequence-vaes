from transformers import AutoTokenizer, AutoModelForMaskedLM, PreTrainedTokenizerFast, DataCollatorForLanguageModeling
import torch
import torch.nn as nn
import sys
import numpy as np
sys.path.append("../VAE_standard")
from models import DNADataset, ALPHABET, SEQ_LENGTH, LATENT_DIM, VAE

from matplotlib import pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
from matplotlib.patches import FancyBboxPatch
from highlight_text import fig_text, ax_text

sys.path.append("..")
import utils

import Bio.Data.CodonTable

from devinterp.utils import (
    EvaluateFn,
    EvalResults,
)

from BIF_sampler import (
    BIFEstimator,
    estimate_bif
)

import copy

MAX_TOKEN_LENGTH = 510
BATCH_SIZE=60
num_masks = 3
TEST_SEQ = 1
TRAIN_CUTOFF = 3000
TEST_TOKEN = 0
DEVICE = "cuda"


def get_BIF_data(
    tokenizer,
    MAX_TOKEN_LENGTH = MAX_TOKEN_LENGTH,
    BATCH_SIZE = BATCH_SIZE,
    num_masks = num_masks,
    TEST_SEQ = TEST_SEQ,
    TRAIN_CUTOFF = TRAIN_CUTOFF,
    TEST_TOKEN = TEST_TOKEN,
    DEVICE = DEVICE):
    """
    returns the requisite data variables needed to run the Bayesian Influence Function

    returns:
    - bif_dataloader: (torch.utils.Data.Dataloader) Given some test seq X, the bif_dataloader returns a dataloader with length len(X) // num_masks. Element i of the dataloader is X with amino acids i:(i+num_masks) masked -- this is used to compute the pairwise influence between every length num_masks pair of AAs and the num_masks AA at location TEST_TOKEN

    - sgld_dataloader: (torch.utils.Data.Dataloader) Dataloader of training data which SGLD uses to update parameters at each SGLD step
    - test_seq: (torch.tensor) index of test seq used for bif_dataloader -- for this sequence, the pairwise influence will be calculated for every num_masks pair of AA with TEST_TOKEN:(TEST_TOKEN+num_masks). Also, the TEST_TOKEN:(TEST_TOKEN + num_masks) is masked here
    
    - test_seq_label: (torch.tensor) the ground truth for test_seq -- test_seq has AA TEST_TOKEN:(TEST_TOKEN + num_masks) masked and this var contains the ground truth labels for the mask

    - test_seq_disp: TEST_SEQ used for displaying influence
    """

    print("MAX_TOKEN_LENGTH=%d\nBATCH_SIZE=%d\nnum_masks=%d\nTEST_SEQ=%d\nTRAIN_CUTOFF=%d\nTEST_TOKEN=%d\nDEVICE=%s"%(MAX_TOKEN_LENGTH, BATCH_SIZE, num_masks, TEST_SEQ, TRAIN_CUTOFF, TEST_TOKEN, DEVICE))

    ## EXTRACTING DATA
    ## ---------------------------
    dataset = DNADataset(f"../data/training_spike.fasta")
    sequences = [utils.get_genome(np.dot(x[0], np.arange(len(ALPHABET)))) for x in dataset]
    print("done extracting sequences!")

    x = Bio.Data.CodonTable.standard_dna_table
    str_seqs = ["".join(x).replace("-","") for x in sequences]
    codons = [[x[num_masks * i:num_masks * (i+1)] for i in range(len(x) // num_masks)] for x in str_seqs]
    aa_drop_na = [[s for s in "".join([x.forward_table.get(s,"") for s in seq][:MAX_TOKEN_LENGTH])] for seq in codons]
    print("done extracting AAs!")

    test_seq = aa_drop_na[TEST_SEQ-1:TEST_SEQ]
    masked_seqs = [seq.copy() for j in range(MAX_TOKEN_LENGTH // num_masks) for seq in test_seq]
    labeled_seqs = [seq for j in range(MAX_TOKEN_LENGTH // num_masks) for seq in test_seq]
    print("done creating element copies!")

    for i in range(len(masked_seqs)):
        j = i % (MAX_TOKEN_LENGTH // num_masks)
        for k in range(num_masks * j, num_masks * (j+1)):
            masked_seqs[i][k] = "<mask>"

    masked_seqs = ["".join(x) for x in masked_seqs]
    print("done creating bif masks!")

    labeled_seqs = ["".join(seq) for seq in labeled_seqs]
    print("done creating bif labels!")

    train_data = ["".join(seq) for seq in aa_drop_na[:TRAIN_CUTOFF]]
    print("done creating sgld dataset!")

    print("masked seqs len: %d"%len(masked_seqs))
    print("length train data: %d"%len(train_data))


    ## PREPARING DATA FOR MODEL
    ## ---------------------------
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15, return_tensors="pt")
    sgld_dataset = tokenizer(text=train_data, return_tensors="pt", add_special_tokens=False, truncation=False, padding=True)["input_ids"]
    sgld_inputs, sgld_labels = data_collator.torch_mask_tokens(sgld_dataset)

    sgld_inputs = sgld_inputs.to(DEVICE)
    sgld_labels = sgld_labels.to(DEVICE)

    print("SGLD inputs, labels")
    print(sgld_inputs.shape)
    print(sgld_labels.shape, "\n")


    bif_inputs = tokenizer(text=masked_seqs, return_tensors="pt", add_special_tokens=False, truncation=False)
    bif_labels = tokenizer(text=labeled_seqs, return_tensors="pt", add_special_tokens=False, truncation=False)["input_ids"]
    bif_labels = torch.where(bif_inputs["input_ids"] == tokenizer.mask_token_id, bif_labels, -100)

    bif_labels = bif_labels.to(DEVICE)
    bif_inputs["input_ids"] = bif_inputs["input_ids"].to(DEVICE)
    bif_inputs["attention_mask"] = bif_inputs["attention_mask"].to(DEVICE)

    print("BIF inputs, labels")
    print(bif_inputs["input_ids"].shape)
    print(bif_labels.shape, "\n")


    test_seq = tokenizer(text=masked_seqs[TEST_TOKEN], return_tensors="pt", add_special_tokens=False, truncation=False)
    test_seq_label = tokenizer(text=labeled_seqs[TEST_TOKEN], return_tensors="pt", add_special_tokens=False, truncation=False)["input_ids"]

    test_seq_label = test_seq_label.squeeze().to(DEVICE)
    test_seq["input_ids"] = test_seq["input_ids"].to(DEVICE)
    test_seq["attention_mask"] = test_seq["attention_mask"].to(DEVICE)

    
    bif_dataloader = torch.utils.data.DataLoader(list(zip(zip(bif_inputs["input_ids"], bif_inputs["attention_mask"]), bif_labels)), batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    sgld_dataloader = torch.utils.data.DataLoader(list(zip(sgld_inputs, sgld_labels)), batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    test_seq_disp = "".join(aa_drop_na[TEST_SEQ])
    test_seq_disp = ["<%s>"%(test_seq_disp[num_masks * i : num_masks * (i+1)]) for i in range(len(test_seq_disp) // num_masks)]

    return {"bif_dataloader":bif_dataloader, "sgld_dataloader":sgld_dataloader, "test_seq":test_seq, "test_seq_label":test_seq_label, "test_seq_disp":test_seq_disp}


def disp_BIF_influence(test_seq, computed_influences, TEST_TOKEN=TEST_TOKEN):
    print("TEST_TOKEN=%d"%TEST_TOKEN)
    NUM_BINS = 100
    DISP_BATCH = 20
    cmap = mpl.colormaps['RdBu_r']
    colors = cmap(np.linspace(0, 1, NUM_BINS))

    def rgb_to_hex(r, g, b):
        return "#{:02X}{:02X}{:02X}".format(int(r*255), int(g*255), int(b*255))

    def normalize(arr, num_bins=NUM_BINS):
        norm_arr = arr / (2*np.max(arr)) + 0.5
        bin_arr = np.digitize(norm_arr, np.linspace(0, 1, num_bins-1))
        return bin_arr

    fig,arr = plt.subplots(1,1,figsize=(20,6))
    arr.axis((0,5,0,5))
    i = 0
    norm_influences = normalize(computed_influences)

    ax_text(
            x=0,  # position on x-axis
            y=5,  # position on y-axis
            s=test_seq[TEST_TOKEN],
            fontsize=20,
            ax=arr,
            highlight_textprops=[
                {"bbox": {"edgecolor": "#969696",
                          "facecolor": "#EEEEEE",
                          "linewidth": 1,
                          "pad": 1}}
            ]
        )

    while (i * 20 < len(test_seq)):
        subseq = test_seq[i * DISP_BATCH : (i + 1) * DISP_BATCH]
        influences = norm_influences[i * DISP_BATCH : (i + 1) * DISP_BATCH]
        ax_text(
            x=0,  # position on x-axis
            y=4.4 - 0.4 * i,  # position on y-axis
            s=" ".join(subseq),
            fontsize=20,
            ax=arr,
            highlight_textprops=[
                {"bbox": {"edgecolor": "#FFFFFF",
                          "facecolor": colors[influences[i]],
                          "linewidth": 0,
                          "pad": 1}} for i in range(len(subseq))
            ]
        )
        i += 1
        if i > 10:
            break
    arr.set_axis_off()
    fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(np.min(computed_influences), np.max(computed_influences)), cmap='RdBu_r'), ax=arr)
    plt.show()

