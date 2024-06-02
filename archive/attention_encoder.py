import torch
import torch.nn as nn
import torch.nn.functional as F


class ReducedSelfAttention(nn.Module):
    def __init__(self, hidden_size: int, sequence_size: int, reduced_hidden_size: int, num_heads: int, intermediate_hidden_size: int = 16):
        super(ReducedSelfAttention, self).__init__()
        self.hidden_size = hidden_size
        self.sequence_size = sequence_size
        self.reduced_hidden_size = reduced_hidden_size

        # Query, Key, and Value transformations
        # self.query = nn.Linear(
        #    hidden_size, intermediate_hidden_size, bias=False)
        self.key = nn.Linear(hidden_size, intermediate_hidden_size, bias=False)
        self.value = nn.Linear(
            hidden_size, intermediate_hidden_size, bias=False)

        self.attention = nn.MultiheadAttention(
            hidden_size, kdim=intermediate_hidden_size, vdim=intermediate_hidden_size, num_heads=num_heads)

        # Learnable matrix to reduce dimensionality of the output sequence
        self.reduce_dim = nn.Linear(
            sequence_size, reduced_hidden_size, bias=False)

    def forward(self, x):

        # Transform inputs to query, key, and value vectors
        keys = self.key(x)
        values = self.value(x)

        # Apply attention weights to the values
        output, _ = self.attention(x, keys, values, need_weights=False)

        # Reduce the sequence dimension
        reduced_output = self.reduce_dim(output.transpose(
            1, 2)).transpose(1, 2)  # Transpose for reduction and then back

        return reduced_output  # [batch_size, reduced_size, hidden_size]
