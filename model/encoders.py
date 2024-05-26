
import torch.nn as nn
import torch
import torch.nn.functional as F
from model.embeddings import SurveyEmbeddings
from model.layers import CustomTabTransformerConv
from torch_frame.nn.conv import TabTransformerConv, ExcelFormerConv
from torch_frame.nn.decoder import ExcelFormerDecoder


class ExcelEncoder(nn.Module):
    """
    Encoder that models the interaction between the columns of the SurveyEmbeddings
    """

    def __init__(self, vocab_size,
                 sequence_len: int,
                 embedding_size: int,
                 output_size: int,
                 num_layers: int = 3,
                 num_heads: int = 4,
                 num_years: int = 14,
                 dropout: float = 0.1,
                 decoder_output: int = 1,
                 layer_type: str = "attn") -> None:
        """
        Args:
            layer_type: either use convolution of attention to model the interactions
        """
        super().__init__()
        assert layer_type in ["excel", "attn", "mixture"], "Wrong layer type"
        self.num_cols = sequence_len
        self.embedding = SurveyEmbeddings(
            vocab_size=vocab_size, n_questions=self.num_cols, n_years=num_years, embedding_dim=embedding_size)

        self.encoders = nn.ModuleList([
            nn.Sequential(
                ExcelFormerConv(channels=embedding_size,
                                num_cols=self.num_cols,
                                num_heads=num_heads,
                                diam_dropout=dropout,
                                aium_dropout=dropout,
                                residual_dropout=dropout),
                nn.Mish(),
                nn.InstanceNorm1d(sequence_len)
            )
            for _ in range(num_layers)])

        self.decoder = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embedding_size, decoder_output),
        )
        # this layer just aggregates the representation of column into one embedding
        self.flatten = ExcelFormerDecoder(in_channels=embedding_size,
                                          out_channels=output_size,
                                          num_cols=self.num_cols)
        self.flatten.activation = nn.Mish()

    def forward(self, year, seq):
        """
        Method that returns full encoding-decoding
        """
        assert seq.size(1) == self.num_cols, "Wrong shapes"
        x = self.embedding(year, seq)
        for encoder in self.encoders:
            x = encoder(x)
        x = self.decoder(x)
        return x

    def get_embedding(self, year, seq):
        """
        Returns initial embeddings of the sequence
        """
        return self.embedding(year, seq)

    def get_encoding(self, year, seq):
        """
        Method that return the embedding of the survey
        """
        x = self.embedding(year, seq)
        for encoder in self.encoders:
            x = encoder(x)
        x = self.flatten(x)
        return x
