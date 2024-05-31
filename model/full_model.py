# Data packages
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, average_precision_score, f1_score, recall_score, precision_score, matthews_corrcoef

from model.rnn import GRUDecoder
from model.encoders import CustomExcelFormer
from data_processing.pipeline import encoding_pipeline, get_generic_name


import matplotlib.pyplot as plt
from model.utils import get_device
from model.dataset import PretrainingDataset
from model.dataset import FinetuningDataset


device = get_device()

# GLOBAL VARIABLES
BATCH_SIZE = 16
HIDDEN_SIZE = 48
ENCODING_SIZE = 48
NUM_HEADS = 4
NUM_LAYERS = 5
NUM_EPOCHS = 12
DETECT_ANOMALY = False
SEQ_LEN = 149
VOCAB_SIZE = 860
LR = 1e-2

assert HIDDEN_SIZE % NUM_HEADS == 0, "Check that the hidden size is divisible"


class PreFerPredictor(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.encoder = CustomExcelFormer(vocab_size=VOCAB_SIZE,
                                         hidden_size=HIDDEN_SIZE,
                                         out_size=ENCODING_SIZE,
                                         n_years=14,
                                         num_heads=NUM_HEADS,
                                         num_layers=NUM_LAYERS,
                                         sequence_len=SEQ_LEN,
                                         aium_dropout=0.3,
                                         diam_dropout=0.1,
                                         residual_dropout=0.1,
                                         embedding_dropout=0.25).to(device)
        self.decoder = GRUDecoder(
            input_size=ENCODING_SIZE,
            hidden_size=HIDDEN_SIZE,
            num_layers=2,
            max_seq_len=14,
            dropout=0.25,

            bidirectional=True,
            with_attention=True
        ).to(device)

        self.enc_dropout = nn.Dropout(0.1)
        self.enc_dropout1d = nn.Dropout1d(0.1)

    def forward(self, input_year, input_seq, labels):
        bs, ss = labels.size(0), 14
        input_year = input_year.reshape(-1).to(device)
        input_seq = input_seq.reshape(bs * ss, -1).to(device)

        # , y=labels.unsqueeze(-1).expand(-1, 14).reshape(-1), mixup_encoded=True)
        encodings, _ = self.encoder(input_year, input_seq)
        encodings = encodings.view(bs, ss, -1)
        encodings = self.enc_dropout(encodings)
        encodings = self.enc_dropout1d(encodings)
        mask = ~((input_seq == 101).sum(-1) == SEQ_LEN).view(bs, ss).detach()
        # Forward pass
        out = self.decoder(encodings, mask=mask).flatten()
        return out
