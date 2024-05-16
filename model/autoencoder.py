import torch.nn as nn
import torch.nn.functional as F
from model.layers import ConvEncoderLayer, ConvDecoderLayer, Norm
from model.embeddings import SurveyEmbeddings


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


################
# ARCHIVE
class _AutoEncoder(nn.Module):
    def __init__(self, num_embeddings: int,
                 encoding_dim: int = 16,
                 dropout: float = 0.2) -> None:
        super(_AutoEncoder, self).__init__()

        self.encoding_dim = encoding_dim
        self.embed = nn.Embedding(num_embeddings, 512)
        self.encoder = nn.Sequential(
            nn.Linear(512, 256),
            nn.Mish(),
            nn.LayerNorm(256),
            nn.AlphDropout(p=dropout),
            nn.Linear(256, 128),
            nn.Mish(),
            nn.LayerNorm(128),
            nn.AlphDropout(p=dropout),
            nn.Linear(128, 64),
            nn.Mish(),
            nn.LayerNorm(64),
            nn.AlphDropout(p=dropout),
            nn.Linear(64, 32),
            nn.Mish(),
            nn.LayerNorm(64),
            # nn.AlphDropout(p=dropout),
            nn.Linear(32, encoding_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 32),
            nn.Mish(),
            nn.Linear(32, 64),
            nn.Mish(),
            nn.Linear(64, 128),
            nn.Mish(),
            nn.Linear(128, 256),
            nn.Mish(),
            nn.Linear(256, 512),
        )

    def forward(self, x):
        x = self.embed(x)

        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def get_loss(self, x):
        x_hat = self.forward(x)
        return F.mse_loss(x_hat, self.embed(x))

    def embed_and_encode(self, x):
        x = self.embed(x)
        return self.encode(x)

    def get_encoding_dim(self):
        return self.encoding_dim


class SimpleAutoEncoder(nn.Module):
    def __init__(self, vocab_size, sequence_len, embedding_size) -> None:
        super().__init__()

        self.embedding = SurveyEmbeddings(
            vocab_size, sequence_len, n_years=14, embedding_dim=256)

        self.cls = nn.Sequential(
            nn.Linear(256, vocab_size, bias=False)
        )

        self.encoder = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
        )

        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
        )

    def forward(self, year, seq, encode_only=False):
        x = self.embedding(year, seq)
        xx = self.encoder(x)
        if encode_only:
            return xx  # What is the shape here ?
        xx = self.decoder(xx)
        xx = self.cls(xx)
        return x, xx
