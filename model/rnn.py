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

    def forward(self, x, mask=None):
        """
        Args:
            x: [BATCH_SIZE, MAX_SEQUENCE_LEN, INPUT_SIZE], it is important to concatenate all the existing questionnaire embedding and then pad them
            mask: binary tensor [BATCH_SIZE, MAX_SEQ_LEN] (False stands for the padded element)
        """
        ######
        # this part is specific to RNNs and padded sequences
        lengths = mask.sum(dim=1)
        lengths, sorted_idx = lengths.sort(0, descending=True)
        x = x[sorted_idx]

        packed_x = pack_padded_sequence(x, lengths.cpu(), batch_first=True)
        packed_x, _ = self.gru(packed_x)
        x, _ = pad_packed_sequence(packed_x, batch_first=True)

        _, original_idx = sorted_idx.sort(0)
        x = x[original_idx]
        # (previous) returns the shape [BATCH_SIZE, MAX_SEQ_LEN, HIDDEN_SIZE]
        # RNN section ends
        ######
        x = self.post_gru(x)
        # (previous) returns  the shape [BATCH_SIZE, MAX_SEQ_LEN, HIDDEN_SIZE]
        x = self.attention(x, mask=mask)
        # (previous) returns the shape [BATCH_SIZE, HIDDEN_SIZE]
        x = self.decoder(x)
        # (previous) returns the shape [BATCH_SIZE, OUTPUT_SIZE]
        return x
