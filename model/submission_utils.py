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


class PreFerPredictor(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.encoder = CustomExcelFormer(vocab_size=860,
                                         hidden_size=48,
                                         out_size=48,
                                         n_years=14,
                                         num_heads=4,
                                         num_layers=5,
                                         sequence_len=149,
                                         aium_dropout=0.3,
                                         diam_dropout=0.1,
                                         residual_dropout=0.1,
                                         embedding_dropout=0.25).to(device)
        self.decoder = GRUDecoder(
            input_size=48,
            hidden_size=48,
            num_layers=2,
            max_seq_len=14,
            dropout=0.25,

            bidirectional=True,
            with_attention=True
        ).to(device)

        self.enc_dropout = nn.Dropout(0.1)
        self.enc_dropout1d = nn.Dropout1d(0.1)
        self.seq_len = 149

    def forward(self, input_year, input_seq, labels):
        bs, ss = labels.size(0), 14
        input_year = input_year.reshape(-1).to(device)
        input_seq = input_seq.reshape(bs * ss, -1).to(device)

        # , y=labels.unsqueeze(-1).expand(-1, 14).reshape(-1), mixup_encoded=True)
        encodings, _ = self.encoder(input_year, input_seq)
        encodings = encodings.view(bs, ss, -1)
        encodings = self.enc_dropout(encodings)
        encodings = self.enc_dropout1d(encodings)
        mask = ~((input_seq == 101).sum(-1) ==
                 self.seq_len).view(bs, ss).detach()
        # Forward pass
        out = self.decoder(encodings, mask=mask).flatten()
        return out

    def predict(self, dataloader, device):
        preds = []
        for batch in dataloader:
            inputs, labels = batch
            labels = labels.to(torch.float).to(device)
            input_year, input_seq = inputs
            # Model
            output = self.forward(input_year=input_year,
                                  input_seq=input_seq, labels=labels)
            probs = F.sigmoid(output).flatten()

            preds.extend(probs.detach().cpu().numpy().tolist())

        return torch.tensor(preds).flatten().numpy()


class DataProcessor:
    def __init__(self,
                 data: pd.DataFrame,
                 outcomes: pd.DataFrame,
                 n_cols: int,
                 codebook_path: str = 'data/codebooks/PreFer_codebook.csv',
                 importance_path: str = 'features_importance_all.csv') -> None:
        self.data = data
        self.outcomes = outcomes
        self.codebook = pd.read_csv(codebook_path)
        self.col_importance = pd.read_csv(importance_path)
        self.n_cols = n_cols

    def convert_to_sequences(self, use_codebook: bool = True):
        self.custom_pairs = self.col_importance.feature.map(
            lambda x: get_generic_name(x)).unique()[:self.n_cols]

        self.sequences = encoding_pipeline(self.data, self.codebook,
                                           custom_pairs=self.custom_pairs,
                                           importance=self.col_importance,
                                           use_codebook=use_codebook)
        self.__preprocessing_pipeline__()

    def make_predictions(self, df: pd.DataFrame, batch_size: int, use_codebook: bool = True):
        self.prediction_sequences = encoding_pipeline(self.data, self.codebook,
                                           custom_pairs=self.custom_pairs,
                                           importance=self.col_importance,
                                           use_codebook=use_codebook)
        
        person_ids = df['nomem_encr'].values
        data_obj = {person_id: (
            torch.tensor(
                [year-2007 for year, _ in wave_responses.items()]).to(device),
            torch.tensor(
                [wave_response for _, wave_response in wave_responses.items()]).to(device)
        )
            for person_id, wave_responses in self.sequences.items()
        }

        # split data based on the splits made for the target
        full_data = {person_id: data_obj[person_id]
                     for person_id in person_ids}

        self.prediction_dataset = PredictionDataset(full_data)
        self.prediction_dataloader = DataLoader(
            self.full_dataset,
            batch_size=batch_size,
            shuffle=False)

        

    def __preprocessing_pipeline__(self):
        self.pretrain_dataset = PretrainingDataset(self.sequences)
        self.seq_len = self.pretrain_dataset.get_seq_len()
        self.vocab_size = self.pretrain_dataset.get_vocab_size()

    def make_traindata(self, batch_size: int):
        """Create dataloader for the whole finetuning dataset.
        At the end creates training dataset and training dataloader
        """
        # Do we still need this filtering?
        outcomes = self.outcomes[self.outcomes.new_child.notna()]
        person_ids = outcomes['nomem_encr'].values
        data_obj = {person_id: (
            torch.tensor(
                [year-2007 for year, _ in wave_responses.items()]).to(device),
            torch.tensor(
                [wave_response for _, wave_response in wave_responses.items()]).to(device)
        )
            for person_id, wave_responses in self.sequences.items()
        }

        # split data based on the splits made for the target
        full_data = {person_id: data_obj[person_id]
                     for person_id in person_ids}

        self.full_dataset = FinetuningDataset(full_data, targets=outcomes)
        self.full_dataloader = DataLoader(
            self.full_dataset,
            batch_size=batch_size,
            shuffle=True)

