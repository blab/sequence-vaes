import os, sys
import argparse
import pandas as pd
import json
import numpy as np
import torch

sys.path.append("..")
from models import VAE, DNADataset, ALPHABET, SEQ_LENGTH, LATENT_DIM

sys.path.append("../../EVE")
from EVE import VAE_model, VAE_encoder, VAE_decoder
from utils import data_utils

class DummyData:
    def __init__(self):
        print("using dummy data class")

        BATCH_SIZE = 64
        EPOCHS = 30
        dataset = DNADataset("../data/training/training_aligned.fasta")
        training_dataset = np.array([dataset[x][0].cpu().numpy() for x in range(len(dataset))])
        del dataset

        self.num_sequences = training_dataset.shape[0]
        self.seq_len = training_dataset.shape[1]
        self.alphabet_size = training_dataset.shape[-1]
        self.Neff = training_dataset.shape[0]

        self.one_hot_encoding = training_dataset
        self.weights = np.ones(training_dataset.shape[0])

        dataset = DNADataset("../data/valid/valid_aligned.fasta")
        valid_dataset = np.array([dataset[x][0].cpu().numpy() for x in range(len(dataset))])
        del dataset

        self.validation_set = valid_dataset
        self.validation_weights = np.ones(valid_dataset.shape[0])
        
if __name__ == '__main__':

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_default_device(device)
    data = DummyData()

    
    enc_params = {
        # "hidden_layers_sizes"               :   [2000,1000,300],      # takes too much memory
        "hidden_layers_sizes"               :   [1000,500,200],
        "z_dim"                             :   50,
        "convolve_input"                    :   False,
        "convolution_input_depth"           :   40,
        "nonlinear_activation"              :   "relu",
        "dropout_proba"                     :   0.0,
        'seq_len'                           :   data.seq_len,
        'alphabet_size'                     :   data.alphabet_size,
        'device'                            :   device
    }

    dec_params = {
        # "hidden_layers_sizes"               :   [300,1000,2000],      # takes too much memory
        "hidden_layers_sizes"               :   [200,500,1000],
        "z_dim"                             :   50,
        "bayesian_decoder"                  :   True,
        "first_hidden_nonlinearity"         :   "relu", 
        "last_hidden_nonlinearity"          :   "relu", 
        "dropout_proba"                     :   0.1,
        "convolve_output"                   :   True,
        "convolution_output_depth"          :   40, 
        "include_temperature_scaler"        :   True, 
        "include_sparsity"                  :   False, 
        "num_tiles_sparsity"                :   0,
        "logit_sparsity_p"                  :   0,
        'seq_len'                           :   data.seq_len,
        'alphabet_size'                     :   data.alphabet_size,
        'device'                            :   device
    }
    

    model_name = "Covid_model1"
    print("Model name: "+str(model_name))
    model = VAE_model.VAE_model(
                    model_name=model_name,
                    data=data,
                    encoder_parameters=enc_params,
                    decoder_parameters=dec_params,
                    # random_seed=args.seed
                    random_seed = 42,
                    device=device
    )

    print(model.device)
    print(model.decoder.device)
    print(model.encoder.device)

    print("Starting to train model: " + model_name)

    training_params = {
        "num_training_steps"                :   400000,
        "learning_rate"                     :   1e-4,
        "batch_size"                        :   256,
        "annealing_warm_up"                 :   0,
        "kl_latent_scale"                   :   1.0,
        "kl_global_params_scale"            :   1.0,
        "l2_regularization"                 :   0.0,
        "use_lr_scheduler"                  :   False,
        "use_validation_set"                :   True,
        "validation_set_pct"                :   0.2,
        "validation_freq"                   :   1000,
        "log_training_info"                 :   True,
        "log_training_freq"                 :   1000,
        "save_model_params_freq"            :   20000,
        'training_logs_location'            :   "./model_checkpoints",
        'model_checkpoint_location'         :   "./model_checkpoints"
    }
    model.train_model(data=data, training_parameters=training_params)
    
    print("Saving model: " + model_name)
    model.save(model_checkpoint=training_params['model_checkpoint_location']+os.sep+model_name+"_final", 
                encoder_parameters=enc_params, 
                decoder_parameters=dec_params,
                training_parameters=training_params
    )
