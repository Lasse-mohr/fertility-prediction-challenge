import torch.nn as nn


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

        return x

  
class SqueezeExcitation(nn.Module):
    """
    Based on the idea of between channel interactions:
    https://arxiv.org/abs/1709.01507
    """

    def __init__(self, num_columns, hidden_size, reduction_ratio=16):
        super(SqueezeExcitation, self).__init__()
        # Assuming the reduction happens across the hidden_size
        self.num_columns = num_columns
        self.reduced_size = max(1, hidden_size // reduction_ratio)

        self.squeeze = nn.AdaptiveAvgPool1d(1)  # Squeezes hidden_size to 1
        self.excitation = nn.Sequential(
            nn.Linear(num_columns, self.reduced_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.reduced_size, num_columns),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: [batch_size, num_columns, hidden_size]
        # Squeeze operation
        # Change to [batch_size, hidden_size, num_columns] for pooling
        z = self.squeeze(x)
        # z shape: [batch_size, hidden_size, 1]

        # Excitation operation
        z = z.view(z.size(0), -1)  # Flatten [batch_size, hidden_size]
        s = self.excitation(z)  # [batch_size, num_columns]
        # Reshape to [batch_size, num_columns, 1] to match original dimensions
        s = s.view(s.size(0), s.size(1), 1)

        return x * s  # Apply recalibration weights to the original input

