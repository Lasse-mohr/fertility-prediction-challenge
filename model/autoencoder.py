import torch
import torch.nn as nn
import torch.nn.functional as F

# partly inspired by https://github.com/chrisvdweth/ml-toolkit/blob/master/pytorch/models/text/classifier/cnn.py


class ConvEncoderLayer(nn.Module):
    def __init__(self, input_size: int,
                 output_size: int,
                 kernel_size: int,
                 padding_len: int,
                 pooling_size: int,
                 dropout: float = 0.1):
        super(ConvEncoderLayer, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.kernel_size = kernel_size
        self.padding_len = padding_len
        self.pooling_size = pooling_size

        self.layer = nn.Conv1d(
            in_channels=input_size,
            out_channels=output_size,
            kernel_size=kernel_size,
            padding=padding_len)
        self.act = nn.Mish()
        self.dropout = nn.Dropout(p=dropout)
        self.norm = nn.LazyInstanceNorm1d()
        self.pool = nn.MaxPool1d(pooling_size)

    def forward(self, x):
        x = self.layer(x)
        x = self.act(x)
        x = self.norm(x)
        x = self.dropout(x)
        x = self.pool(x)
        # print(x.shape)
        return x

    def get_output_shape(self, x):
        sc = (x + 2 * self.padding_len - self.kernel_size) + 1
        output = (sc - self.pooling_size)/self.pooling_size + 1
        return int(output)


class ConvDecoderLayer(nn.Module):
    def __init__(self, input_size: int,
                 output_size: int,
                 hidden_dim: int,
                 kernel_size: int = 24,
                 dilation: int = 4,
                 dropout: float = 0.1):
        super(ConvDecoderLayer, self).__init__()

        self.layer = nn.ConvTranspose1d(in_channels=input_size,
                                        out_channels=output_size,
                                        kernel_size=kernel_size,
                                        dilation=dilation)
        self.act = nn.Mish()
        self.dropout = nn.Dropout(p=dropout)
        self.norm = nn.LazyInstanceNorm1d()
        self.ff = nn.LazyLinear(hidden_dim)

    def forward(self, x):
        x = self.layer(x)
        x = self.act(x)
        x = self.norm(x)
        x = self.dropout(x)
        x = self.ff(x)
        # print(x.shape)
        return x


class AutoEncoder(nn.Module):
    def __init__(self, vocab_size: int,
                 sequence_len: int,
                 embedding_size: int,
                 encoding_size: int) -> None:
        super(AutoEncoder, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.encoding_size = encoding_size
        self.sequence_len = sequence_len

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.encoder = nn.Sequential(
            ConvEncoderLayer(input_size=embedding_size, output_size=embedding_size //
                             4, kernel_size=3, padding_len=1, pooling_size=4),
            ConvEncoderLayer(input_size=embedding_size // 4, output_size=embedding_size //
                             8, kernel_size=6, padding_len=1, pooling_size=4),
            ConvEncoderLayer(input_size=embedding_size // 8, output_size=encoding_size,
                             kernel_size=6, padding_len=1, pooling_size=4),
            nn.Mish(),
            nn.AdaptiveAvgPool1d(1)
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

    def get_encoder_shapes(self):
        s = self.sequence_len
        out = []
        for encoder in self.encoder[:3]:
            out.append(encoder.get_output_shape(s))
            s = out[-1]
        return out

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)  # we need to switch the things around
        # significantly reduce the dimensionality while allowing for interactions between 2D dimensions
        x = self.encoder(x)

        # Decoding
        x = self.decoder(x)
        # Reshape to match the expected input for ConvTranspose1d
        # Switch dimensions back to [batch_size, seq_len, embedding_size]
        x = x.permute(0, 2, 1)
        return x

    def encode(self, x):
        """
        Return the embedding of a survey
        """
        x = self.embedding(x)
        x = x.permute(0, 2, 1)  # we need to switch the things around
        # significantly reduce the dimensionality while allowing for interactions between 2D dimensions
        x = self.encoder(x).view(x.size(0), -1)
        return x

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
