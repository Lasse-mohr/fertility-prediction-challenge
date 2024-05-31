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
from model.dataset import FinetuningDataset, PredictionDataset


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

    def predict(self, dataloader):
        preds = []
        for batch in dataloader:
            inputs, labels = batch
            labels = labels.to(torch.float).to(self.device)
            input_year, input_seq = inputs
            ### Model
            output = self.forward(input_year=input_year, input_seq=input_seq, labels=labels)
            probs = F.sigmoid(output).flatten()

            preds.extend(probs.detach().cpu().numpy().tolist())

        return torch.tensor(preds).flatten().numpy()


class DataClass:
    def __init__(self,
                 data_path: str = "data/training_data/PreFer_train_data.csv",
                 targets_path: str = 'data/training_data/PreFer_train_outcome.csv',
                 codebook_path: str = 'data/codebooks/PreFer_codebook.csv',
                 importance_path: str = 'features_importance_all.csv',
                 to_predict_df: str = None) -> None:
        self.data = pd.read_csv(data_path, low_memory=False)
        self.targets = pd.read_csv(targets_path)
        self.codebook = pd.read_csv(codebook_path)
        self.col_importance = pd.read_csv(importance_path)
        if to_predict_df is not None:
            self.prediction_data = to_predict_df
    def make_sequences(self, n_cols: int, use_codebook: bool = True):
        custom_pairs = self.col_importance.feature.map(lambda x: get_generic_name(x)).unique()[:n_cols]
        self.custom_pairs = custom_pairs
        self.sequences = encoding_pipeline(self.data, self.codebook, 
                                           custom_pairs=custom_pairs, 
                                           importance=self.col_importance, 
                                           use_codebook=use_codebook)
    def make_pretraining(self):
        self.pretrain_dataset = PretrainingDataset(self.sequences)
        self.seq_len = self.pretrain_dataset.get_seq_len()
        self.vocab_size = self.pretrain_dataset.get_vocab_size()

    
    def make_prediction_sequences(self):
        self.prediction_sequences = encoding_pipeline(self.data, self.codebook, 
                                           custom_pairs=self.custom_pairs, 
                                           importance=self.col_importance, 
                                           use_codebook=False)


    def prepare_prediction(self, batch_size):
        """Create dataloader for the whole finetuning dataset"""
        self.make_prediction_sequences()
        dataset = {person_id: (
                torch.tensor([year-2007 for year, _ in wave_responses.items()]).to(device),
                torch.tensor([ wave_response for _, wave_response in wave_responses.items()]).to(device)
                )
                for person_id, wave_responses in self.prediction_sequences.items()
                }

   
        self.prediction_dataset = PredictionDataset(dataset)
        self.prediction_dataloader = DataLoader(self.prediction_dataset, batch_size=batch_size, shuffle=False)


    def make_full_pretraining(self, batch_size): 
        """Create dataloader for the whole finetuning dataset"""
        targets = self.targets[self.targets.new_child.notna()]
        full_person_ids =  targets['nomem_encr'].values
        rnn_data = {person_id: (
                torch.tensor([year-2007 for year, _ in wave_responses.items()]).to(device),
                torch.tensor([ wave_response for _, wave_response in wave_responses.items()]).to(device)
                )
                for person_id, wave_responses in self.sequences.items()
                }

        # split data based on the splits made for the target
        full_data = {person_id: rnn_data[person_id] for person_id in full_person_ids}
        self.full_dataset = FinetuningDataset(full_data, targets = targets)
        self.full_dataloader = DataLoader(self.full_dataset, batch_size=batch_size, shuffle=True)


    def make_finetuning(self, batch_size, test_size: float = 0.2, val_size: float = 0.2):
        """
        Create dataloaders for the train/val/test splits.
        """
        targets = self.targets[self.targets.new_child.notna()]
        train_person_ids, test_person_ids = train_test_split(targets['nomem_encr'], test_size=test_size, random_state=42)
        train_person_ids, val_person_ids = train_test_split(train_person_ids, test_size=val_size, random_state=42)
        rnn_data = {person_id: (
                torch.tensor([year-2007 for year, _ in wave_responses.items()]).to(device),
                torch.tensor([ wave_response for _, wave_response in wave_responses.items()]).to(device)
                )
                for person_id, wave_responses in self.sequences.items()
                }

        # split data based on the splits made for the target
        train_data = {person_id: rnn_data[person_id] for person_id in train_person_ids}
        val_data = {person_id: rnn_data[person_id] for person_id in val_person_ids}
        test_data = {person_id: rnn_data[person_id] for person_id in test_person_ids}

        self.train_dataset = FinetuningDataset(train_data, targets = targets)
        self.val_dataset = FinetuningDataset(val_data, targets = targets)
        self.test_dataset = FinetuningDataset(test_data, targets = targets)
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False)
        self.test_dataloader  = DataLoader(self.test_dataset,  batch_size=batch_size)



