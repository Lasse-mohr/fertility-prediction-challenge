import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvEncoderLayer(nn.Module):
    def __init__(self, input_size: int, output_size: int, dropout: float = 0.1):
        super(ConvEncoderLayer, self).__init__()
        self.layer = nn.Conv1d(
            in_channels=input_size, out_channels=output_size, kernel_size=3, padding=1)
        self.act = nn.Mish()
        self.norm = nn.LayerNorm([output_size])
        self.dropout = nn.Dropout(p=dropout)
        self.pool = nn.MaxPool1d(2)

    def forward(self, x):
        x = self.layer(x)
        x = self.act(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.pool(x)
        return x


class ConvDecoderLayer(nn.Module):
    def __init__(self, input_size: int, output_size: int, normalize: bool = True, dropout: float = 0.1):
        super(ConvEncoderLayer, self).__init__()
        self.layer = nn.Conv1d(
            in_channels=input_size, out_channel=output_size, kernel_size=3, padding=1)
        self.act = nn.Mish()
        if normalize:
            self.norm = nn.LayerNorm(output_size)
        else:
            self.norm = None

        if dropout is not None:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None


class AutoEncoderLayer(nn.Module):
    def __init__(self, input_size: int, output_size: int, normalize: bool = True, dropout: float = 0.1):
        super(AutoEncoderLayer, self).__init__()

        self.linear = nn.Linear(input_size, output_size)
        self.act = nn.Mish()
        if normalize:
            self.norm = nn.LayerNorm(output_size)
        else:
            self.norm = None

        if dropout is not None:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None

    def remove_norm(self):
        try:
            del self.norm
        except:
            pass
        self.norm = None

    def remove_dropout(self):
        try:
            del self.dropout
        except:
            pass

        self.dropout = None

    def forward(self, x):
        x = self.linear(x)
        x = self.act(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x


class AutoEncoder(nn.Module):
    def __init__(self, vocab_size: int,
                 embedding_size: int,
                 encoding_size: int,
                 num_layers: int,
                 dropout: float = 0.2) -> None:
        super(AutoEncoder, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.encoding_size = encoding_size

        self.embedding = nn.Embedding(vocab_size, embedding_size)

        self.layer_sizes = [embedding_size // 2 **
                            i for i in range(num_layers)] + [encoding_size]

        self.encoder = nn.ModuleList([AutoEncoderLayer(input_size=self.layer_sizes[i],
                                                       output_size=self.layer_sizes[i+1], dropout=dropout) for i in range(0, num_layers)])

        self.decoder = nn.ModuleList([AutoEncoderLayer(input_size=self.layer_sizes[i],
                                                       output_size=self.layer_sizes[i-1], dropout=dropout) for i in range(num_layers, 0, -1)])

        # Generaly you don't normalize the last layer of decoder or encoder
        self.encoder[-1].remove_norm()
        self.decoder[-1].remove_norm()
        # + generally you do not do dropout before or at the last layer
        self.decoder[-1].remove_dropout()
        self.decoder[-2].remove_dropout()
        self.encoder[-1].remove_dropout()

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)  # we need to switch the things around
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def get_loss(self, x):
        x_hat = self.forward(x)
        return F.mse_loss(x_hat, self.embed(x))

    def embed_and_encode(self, x):
        x = self.embed(x)
        return self.encoder(x)

    def get_encoding_size(self):
        return self.encoding_size


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
