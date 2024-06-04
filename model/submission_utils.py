# Data packages
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader

from model.rnn import GRUDecoder
from model.encoders import CustomExcelFormer
from data_processing.pipeline import get_generic_name, to_sequences


from model.utils import get_device
from model.dataset import FinetuningDataset, PredictionDataset
import warnings

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

    def predict(self, dataloader, device: str):
        preds = []
        print(len(dataloader))
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
                 cleaned_df: pd.DataFrame,
                 codebook: pd.DataFrame,
                 col_importance: pd.DataFrame,
                 n_cols: int = 150,
                 ) -> None:
        self.data = cleaned_df
        self.codebook = codebook
        self.col_importance = col_importance
        self.n_cols = n_cols

    def make_custom_pairs(self):
        """
        Make a list of ordered ColumnIDs. Order is based on the 
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.questions_to_use = self.col_importance.feature.map(
                lambda x: get_generic_name(x)).unique()[:self.n_cols]

    def df_to_sequences(self, use_codebook: bool = True):
        """
        Converts the training dataframe into sequences (format that can be used by the CustomExcelFormer)
        """
        self.make_custom_pairs()
        self.sequences = to_sequences(df=self.data,
                                      codebook=self.codebook,
                                      use_codebook=use_codebook,
                                      custom_pairs=self.questions_to_use,
                                      importance=self.col_importance)

    def prepare_traindata(self, outcomes: pd.DataFrame,  batch_size: int = 16):
        """Create dataloader for the whole finetuning dataset.
        At the end creates training dataset and training dataloader
        """
        # Do we still need this filtering?
        outcomes = outcomes[outcomes.new_child.notna()]
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

        full_dataset = FinetuningDataset(full_data, targets=outcomes)
        self.full_dataloader = DataLoader(
            full_dataset,
            batch_size=batch_size,
            shuffle=True)

    def prepare_predictdata(self, df: pd.DataFrame, batch_size: int = 16, use_codebook: bool = True):
        """
        Method takes the dataframe (with unseen data) without any target values,
        and assembles a 'prediction_dataloader'.
        """
        self.make_custom_pairs()

        _sequences = to_sequences(df=df,
                                  codebook=self.codebook,
                                  use_codebook=use_codebook,
                                  custom_pairs=self.questions_to_use,
                                  importance=self.col_importance)

        person_ids = df['nomem_encr'].values
        data_obj = {person_id: (
            torch.tensor(
                [year-2007 for year, _ in wave_responses.items()]).to(device),
            torch.tensor(
                [wave_response for _, wave_response in wave_responses.items()]).to(device)
        )
            for person_id, wave_responses in _sequences.items()
        }

        # split data based on the splits made for the target
        data = {person_id: data_obj[person_id]
                for person_id in person_ids}

        prediction_dataset = PredictionDataset(data)
        self.prediction_dataloader = DataLoader(
            prediction_dataset,
            batch_size=batch_size,
            shuffle=False)
