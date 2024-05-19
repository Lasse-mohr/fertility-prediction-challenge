import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class AggAttention(nn.Module):
    """
    Implementation of the Bahdanau Attention
    """

    def __init__(self, hidden_size: int):
        super(AggAttention, self).__init__()
        z = torch.Tensor(hidden_size)
        nn.init.uniform_(z, a=-0.1, b=0.1)  # initialize weights uniformly
        # self.context are the learned attention weights
        self.register_parameter("context", nn.Parameter(z))
        self.act = nn.Softmax(dim=1)

    def forward(self, x, mask=None):
        """
        Args:
            x: tensor [BATCH_SIZE, MAX_SEQ_LEN, HIDDEN_SIZE]
            mask: binary tensor [BATCH_SIZE, MAX_SEQ_LEN] (False stands for the padded element, aka year when the questionnaire is absent)
        """
        scores = torch.einsum("bij, j -> bi", x, self.context)
        # (previous) return [BATCH_SIZE, MAX_SEQ_LEN]
        if mask is not None:
            # mask out the absent elements
            scores = scores.masked_fill(~mask, float("-inf"))
        scores = self.act(scores)
        # (previous) return [BATCH_SIZE, MAX_SEQ_LEN]
        output = torch.einsum("bij, bi -> bj", x, scores)
        # (previous) return [BATCH_SIZE, HIDDEN_SIZE]
        return output


class GRUDecoder(nn.Module):
    """
    GRU Decoder with Additive Attention Mechanism
    """

    def __init__(self,
                 input_size: int = 512,
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 max_seq_len: int = 10,
                 output_size: int = 1,
                 dropout: float = 0.2,
                 bidirectional: bool = True,
                 with_attention: bool = True,
                 xavier_initialization: bool = True) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.max_seq_len = max_seq_len
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        gru_dropout = 0.0 if num_layers == 1 else dropout
        self.post_gru_size = 2 * hidden_size if bidirectional else hidden_size

        self.in_layer = nn.Sequential(
            nn.LazyInstanceNorm1d(),  # maybe do a batch norm?
            nn.Dropout(p=dropout),
            nn.LazyLinear(input_size),
            # nn.Mish(),
            nn.LayerNorm(input_size)
        )

        self.gru = nn.GRU(
            batch_first=True,
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=gru_dropout
        )

        # self.post_gru = nn.Sequential(
        #    nn.Mish(),
        #    nn.LayerNorm(normalized_shape=self.post_gru_size),
        #    nn.Dropout(p=dropout),
        #    nn.Linear(self.post_gru_size, self.hidden_size),
        #    nn.Mish(),
        #    nn.LayerNorm(normalized_shape=self.hidden_size)
        # )

        if with_attention:
            self.aggregation = AggAttention(hidden_size=self.hidden_size)
        else:
            self.aggregation = self.mean

        # Output Layer
        self.decoder = nn.Linear(
            self.hidden_size, self.output_size, bias=False)

        if xavier_initialization:
            self.init_parameters()

        print("The model is going to set all input MASK to None")

    def init_parameters(self):
        """
        Initialize the parameters with Xavier Initialization.
        """
        # nn.init.xavier_uniform_(self.in_layer[2].weight, gain=2/3)
        nn.init.xavier_uniform_(self.decoder.weight, gain=1.0)

    def mean(self, x, mask):
        if mask is not None:
            denom = torch.sum(mask, -1, keepdim=True)
            x = torch.div(torch.sum(x * mask.unsqueeze(-1), dim=1), denom)
        else:
            x = torch.sum(x, dim=1)
        return x

    def forward(self, x, mask=None):
        """
        Args:
            x: [BATCH_SIZE, MAX_SEQUENCE_LEN, INPUT_SIZE], it is important to concatenate all the existing questionnaire embedding and then pad them
            mask: binary tensor [BATCH_SIZE, MAX_SEQ_LEN] (False stands for the padded element)
        """

        # This layer makes sure that dimensions are aligned and normalized
        x = self.in_layer(x)
        # (previous) returns [BATCH_SIZE, MAX_SEQ_LEN, HIDDEN_SIZE]

        if mask is not None:
            x = x * mask.unsqueeze(-1)

        xx, _ = self.gru(x)
        # (previous) returns the shape [BATCH_SIZE, MAX_SEQ_LEN, HIDDEN_SIZE]
        ######
        # xx = self.post_gru(xx)
        # (previous) returns  the shape [BATCH_SIZE, MAX_SEQ_LEN, HIDDEN_SIZE]
        xx = self.aggregation(xx, mask=mask)
        # (previous) returns the shape [BATCH_SIZE, HIDDEN_SIZE]
        xx = self.decoder(xx)
        # (previous) returns the shape [BATCH_SIZE, OUTPUT_SIZE]
        return xx


class SimpleDecoder(nn.Module):
    def __init__(self,
                 input_size: int,
                 output_size: int) -> None:
        super().__init__()
        self.out = nn.Linear(input_size, output_size, bias=False)

    def forward(self, x, mask=None):
        if mask is not None:
            denom = torch.sum(mask, -1, keepdim=True)
            x = torch.div(torch.sum(x * mask.unsqueeze(-1), dim=1), denom)
        else:
            x = torch.sum(x, dim=1)

        return self.out(x)
