import torch.nn as nn
import torch


class AggAttention(nn.Module):
    """
    Implementation of the Bahdanau Attention
    """

    def __init__(self, hidden_size: int):
        super(AggAttention, self).__init__()
        z = torch.Tensor(hidden_size)
        nn.init.uniform_(z, a=-0.1, b=0.1)  # initialize weights uniformly
        self.register_parameter("context", nn.Parameter(z))

        self.act = nn.Softmax(dim=1)

    def forward(self, x):
        scores = torch.einsum("bij, j -> bi", x, self.context)
        # (previous) return [BATCH_SIZE, SEQ_LEN]
        scores = self.act(scores)
        # (previous) return [BATCH_SIZE, SEQ_LEN]
        output = torch.einsum("bij, bi -> bj", x, scores)
        # (previous) return [BATCH_SIZE, HIDDEN_SIZE]
        return output


class GRUDecoder(nn.Module):
    def __init__(self,
                 input_size: int = 512,
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 output_size: int = 1,
                 dropout: float = 0.2,
                 bidirectional: bool = True) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        gru_dropout = 0.0 if num_layers == 1 else dropout
        self.post_gru_size = 2 * hidden_size if bidirectional else hidden_size

        self.gru = nn.GRU(
            batch_first=True,
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=gru_dropout
        )

        self.post_gru = nn.Sequential(
            nn.Mish(),
            nn.LayerNorm(normalized_shape=self.post_gru_size),
            nn.AlphaDropout(p=dropout),
            nn.Linear(self.post_gru_size, self.hidden_size),
            nn.Mish(),
            nn.LayerNorm(normalized_shape=self.hidden_size)
        )

        self.attention = AggAttention(hidden_size=self.hidden_size)
        self.decoder = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x):
        """
        Args:
            x: [BATCH_SIZE, SEQUENCE_LEN, INPUT_SIZE]
        """
        x, _ = self.gru(x)
        #  (previous) returns the shape [BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE]
        x = self.post_gru(x)
        # (previous) returns  the shape [BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE]
        x = self.attention(x)
        # (previous) returns the shape [BATCH_SIZE, HIDDEN_SIZE]
        x = self.decoder(x)
        # (previous) returns the shape [BATCH_SIZE, OUTPUT_SIZE]
        return x
