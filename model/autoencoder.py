import torch.nn as nn
import torch.nn.functional as F
from model.layers import ConvEncoderLayer, ConvDecoderLayer
from model.embeddings import SurveyEmbeddings
from torch_frame.nn.conv import TabTransformerConv, ExcelFormerConv
from torch_frame.nn.decoder import ExcelFormerDecoder


class TabularEncoder(nn.Module):
    """
    Encoder that models the interaction between the columns of the SurveyEmbeddings
    """

    def __init__(self, vocab_size,
                 sequence_len: int,
                 embedding_size: int,
                 output_size: int,
                 num_cols: int,
                 num_layers: int = 3,
                 num_heads: int = 4,
                 dropout: float = 0.1,
                 layer_type: str = "conv") -> None:
        """
        Args:
            layer_type: either use convolution of attention to model the interactions
        """
        super().__init__()
        assert layer_type in ["excel", "conv"], "Wrong layer type"
        self.embedding = SurveyEmbeddings(
            vocab_size, sequence_len, n_years=14, embedding_dim=embedding_size)

        if layer_type == "conv":
            self.encoders = nn.ModuleList([
                nn.Sequential(
                    TabTransformerConv(channels=embedding_size,
                                       num_heads=num_heads,
                                       attn_dropout=dropout,
                                       ffn_dropout=dropout),
                    nn.Mish(),
                    nn.InstanceNorm1d(sequence_len))
                for _ in range(num_layers)])
        elif layer_type == "excel":
            self.encoders = nn.ModuleList([
                nn.Sequential(
                    ExcelFormerConv(channels=embedding_size,
                                    num_cols=num_cols,
                                    num_heads=num_heads,
                                    diam_dropout=dropout,
                                    aium_dropout=dropout,
                                    residual_dropout=dropout),
                    nn.Mish(),
                    nn.InstanceNorm1d(sequence_len)
                )
                for _ in range(num_layers)])

        # this layer just aggregates the representation of column into one embedding
        self.flatten = ExcelFormerDecoder(in_channels=embedding_size,
                                          out_channels=output_size,
                                          num_cols=num_cols)

    def forward(self, year, seq):
        """
        Method that returns full encoding-decoding
        """
        x = self.embedding(year, seq)
        for encoder in self.encoders:
            x = encoder(x)
        x = self.flatten(x)
        return x

    def get_encoding(self, year, seq):
        """
        Method that return the embedding of the survey
        """
        return self.forward(year, seq)


class SimpleAutoEncoder(nn.Module):
    def __init__(self, vocab_size, sequence_len: int, embedding_size: int, dropout = 0.1) -> None:
        super().__init__()

        self.dropout = dropout

        self.embedding = SurveyEmbeddings(
            vocab_size, sequence_len, n_years=14, embedding_dim=embedding_size)

        self.out = nn.Sequential(
            nn.Linear(embedding_size, vocab_size, bias=False)
        )

        self.encoder = nn.Sequential(
            nn.Linear(embedding_size, embedding_size // 2),
            nn.Mish(),
            nn.LayerNorm(embedding_size // 2),
            nn.Dropout(self.dropout),
            nn.Linear(embedding_size // 2, embedding_size // 4),
            nn.Mish(),
            nn.LayerNorm(embedding_size // 4),
            nn.Dropout(self.dropout),
            nn.Linear(embedding_size // 4, embedding_size // 8),
            nn.LayerNorm(embedding_size // 8),
            nn.Dropout(self.dropout),
        )

        self.decoder = nn.Sequential(
            nn.Linear(embedding_size // 8, embedding_size // 4),
            nn.Mish(),
            nn.LayerNorm(embedding_size // 4),
            nn.Dropout(self.dropout),
            nn.Linear(embedding_size // 4, embedding_size // 2),
            nn.Mish(),
            nn.LayerNorm(embedding_size // 2),
            nn.Dropout(self.dropout),
            nn.Linear(embedding_size // 2, embedding_size),
        )

    def forward(self, year, seq):
        """
        Method that returns full encoding-decoding
        """
        embeddings = self.embedding(year, seq)
        x = self.encoder(embeddings)
        x = self.decoder(x)
        x = self.out(x)
        return x

    def get_encoding(self, year, seq):
        """
        Method that return the embedding of the survey
        """
        embeddings = self.embedding(year, seq)
        x = self.encoder(embeddings)
        return x


class AutoEncoder(nn.Module):
    def __init__(self, vocab_size: int,
                 embedding_size: int,
                 encoding_size: int,
                 sequence_len: int = 3268) -> None:
        super().__init__()

        self.embedding = SurveyEmbeddings(
            vocab_size, sequence_len, n_years=14, embedding_dim=embedding_size)

        self.vocab_size = vocab_size
        self.sequence_len = sequence_len
        self.embedding_size = embedding_size
        self.encoding_size = encoding_size

        self.encoder = nn.Sequential(
            ConvEncoderLayer(input_size=embedding_size, output_size=embedding_size //
                             4, kernel_size=3, padding_len=1, pooling_size=4),
            ConvEncoderLayer(input_size=embedding_size // 4, output_size=embedding_size //
                             8, kernel_size=6, padding_len=1, pooling_size=4),
            ConvEncoderLayer(input_size=embedding_size // 8, output_size=encoding_size,
                             kernel_size=6, padding_len=1, pooling_size=4),
            nn.Mish(),
            nn.AdaptiveAvgPool1d(1)  # produces flat embedding of the table
        )

        self.decoder = nn.Sequential(
            ConvDecoderLayer(input_size=encoding_size,
                             output_size=embedding_size // 8,
                             hidden_dim=self.get_encoder_shapes()[-1],
                             kernel_size=24, dilation=4),
            ConvDecoderLayer(input_size=embedding_size // 8,
                             output_size=embedding_size // 4, kernel_size=24, dilation=4,
                             hidden_dim=self.get_encoder_shapes()[-2]),
            ConvDecoderLayer(input_size=embedding_size // 4,
                             output_size=embedding_size, kernel_size=24, dilation=4,
                             hidden_dim=self.get_encoder_shapes()[-3]
                             ),
            nn.Mish(),
            nn.LazyLinear(sequence_len)
        )

        self.cls = nn.Sequential(
            # Norm(),
            nn.Linear(embedding_size, vocab_size, bias=False)
        )
        self.cls[0].weight = self.embedding.answer_embedding.weight

    def get_encoder_shapes(self):
        s = self.sequence_len
        out = []
        for encoder in self.encoder[:3]:
            out.append(encoder.get_output_shape(s))
            s = out[-1]
        return out

    def forward(self, year, seq, encode_only=False):
        x = self.embedding(year, seq)
        x = x.permute(0, 2, 1)  # we need to switch the things around
        # significantly reduce the dimensionality while allowing for interactions between 2D dimensions
        x = self.encoder(x)
        if encode_only:
            return x.view(x.size(0), -1)

        # Decoding
        x = self.decoder(x)
        # Reshape to match the expected input for ConvTranspose1d
        # Switch dimensions back to [batch_size, seq_len, embedding_size]
        x = x.permute(0, 2, 1)
        logits = self.cls(x)
        return x, logits
