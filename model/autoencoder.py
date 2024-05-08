import torch
import torch.nn as nn
import torch.nn.functional as F

# partly inspired by https://github.com/chrisvdweth/ml-toolkit/blob/master/pytorch/models/text/classifier/cnn.py


class ConvEncoderLayer(nn.Module):
    def __init__(self, input_size: int,
                 output_size: int,
                 kernel_size: int,
                 padding_len: int,
                 pooling_size: int):
        super(ConvEncoderLayer, self).__init__()
        self.layer = nn.Conv1d(
            in_channels=input_size,
            out_channels=output_size,
            kernel_size=kernel_size,
            padding=padding_len)
        self.act = nn.Mish()
        self.pool = nn.MaxPool1d(pooling_size)

    def forward(self, x):
        x = self.layer(x)
        x = self.act(x)
        x = self.pool(x)
        print(x.shape)
        return x


class ConvDecoderLayer(nn.Module):
    def __init__(self, input_size: int,
                 output_size: int,
                 conv_kernel_size: int,
                 stride: int,
                 normalize: bool = True,
                 dropout: float = 0.1):
        super(ConvDecoderLayer, self).__init__()
        self.layer = nn.ConvTranspose1d(
            in_channels=input_size,
            out_channels=output_size,
            kernel_size=conv_kernel_size,
            stride=stride,
            padding=1,
            output_padding=1)  # Ensure the output shape is correctly adjusted
        self.act = nn.Mish()

        if normalize:
            self.norm = nn.LazyInstanceNorm1d()
        else:
            self.norm = None

        if dropout is not None:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None

    def forward(self, x):
        x = self.layer(x)
        x = self.act(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.dropout is not None:
            x = self.dropout(x)
        print(x.shape)
        return x


class AutoEncoder(nn.Module):
    def __init__(self, vocab_size: int,
                 embedding_size: int,
                 encoding_size: int) -> None:
        super(AutoEncoder, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.encoding_size = encoding_size

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.conv_encoders = nn.Sequential(ConvEncoderLayer(input_size=embedding_size, output_size=256, conv_kernel_size=2, pool_kernel_size=2),
                                           ConvEncoderLayer(
                                               input_size=256, output_size=128, conv_kernel_size=3, pool_kernel_size=2),
                                           ConvEncoderLayer(
                                               input_size=128, output_size=64, conv_kernel_size=3, pool_kernel_size=2),
                                           ConvEncoderLayer(input_size=64, output_size=8, conv_kernel_size=3, pool_kernel_size=2))
        self.linear_encoder = nn.Sequential(nn.LazyLinear(out_features=encoding_size),
                                            nn.Mish(),
                                            nn.LayerNorm(encoding_size),
                                            nn.Linear(encoding_size, encoding_size))  # I am using Lazy layer since it is non-trivial to calculate the output of convolutions
        # the model will figure it out
        # Decoder
        self.linear_decoder = nn.Sequential(
            # Prepare to reshape into [batch, channels, length]
            nn.Linear(encoding_size, encoding_size * 4),
            nn.Mish()
        )
        self.conv_decoders = nn.Sequential(
            ConvDecoderLayer(input_size=self.encoding_size,
                             output_size=64, conv_kernel_size=3, stride=2),
            ConvDecoderLayer(input_size=64, output_size=128,
                             conv_kernel_size=3, stride=2),
            ConvDecoderLayer(input_size=128, output_size=256,
                             conv_kernel_size=3, stride=2),
            ConvDecoderLayer(
                input_size=256, output_size=embedding_size, conv_kernel_size=6, stride=4)
        )

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)  # we need to switch the things around
        # significantly reduce the dimensionality while allowing for interactions between 2D dimensions
        x = self.conv_encoders(x)
        x = x.view(x.size(0), -1)  # flatten the input of convolutions
        x = self.linear_encoder(x)

        # Decoding
        x = self.linear_decoder(x)
        # Reshape to match the expected input for ConvTranspose1d
        x = x.view(x.size(0), self.encoding_size, -1)
        x = self.conv_decoders(x)
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
        x = self.conv_encoders(x)
        x = x.view(x.size(0), -1)  # flatten the input of convolutions
        x = self.linear_encoder(x)
        return x


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
