import torch.nn as nn


class GRUDecoder(nn.Module):
    def __init__(self,
                 input_size: int = 512,
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 output_size: int = 1,
                 dropout: float = 0.3,
                 bidirectional: bool = True) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        gru_dropout = 0.0 if num_layers == 1 else dropout

        self.gru = nn.GRU(
            batch_first=True,
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=gru_dropout
        )
        self.post_gru = nn.ModuleList([nn.Mish(),
                                       nn.LayerNorm(
                                           normalized_shape=hidden_size),
                                       nn.AlphaDropout(p=dropout)]
                                      )
        self.decoder = nn.ModuleList([
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Mish(),
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, self.output_size)]
        )

    def forward(self, x):
        x = self.gru(x)
        x = self.post_gru(x)
        x = self.decoder(x)
        return x
