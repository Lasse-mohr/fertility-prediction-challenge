import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch

from torch_frame.nn.conv import TableConv
from torch_frame.nn.conv.tab_transformer_conv import GEGLU, FFN, SelfAttention
from torch_frame.nn.conv.excelformer_conv import init_attenuated, AiuM


class CustomTabTransformerConv(TableConv):
    r"""
    Mixture of ExcelFormer and TabTransformer

    Args:
        channels (int): Input/output channel dimensionality
        num_heads (int): Number of attention heads
        attn_dropout (float): attention module dropout (default: :obj:`0.`)
        aium_dropout (float): attention module dropout (default: :obj:`0.`)
    """

    def __init__(self, channels: int, num_heads: int, residual_dropout: float = 0.2, attn_dropout: float = 0.,
                 auim_dropout: float = 0.):
        super().__init__()
        self.norm_1 = nn.LayerNorm(channels)
        self.attn = SelfAttention(channels, num_heads, attn_dropout)
        self.norm_2 = nn.LayerNorm(channels)
        self.ffn = AiuM(channels, auim_dropout)
        self.reset_parameters()
        self.register_parameter("alpha", nn.Parameter(
            torch.tensor([0.0]), requires_grad=True))
        self.register_parameter("beta", nn.Parameter(
            torch.tensor([0.0]), requires_grad=True))

        self.rd_1 = nn.Dropout(residual_dropout)
        self.rd_2 = nn.Dropout(residual_dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm_1(x)
        out = self.attn(x)
        x = x + self.alpha * self.rd_1(out)
        x = x + self.beta * self.rd_2(self.ffn(x))
        return x

    def reset_parameters(self):
        self.norm_1.reset_parameters()
        # self.attn.reset_parameters()
        init_attenuated(self.attn.lin_q)
        init_attenuated(self.attn.lin_k)
        init_attenuated(self.attn.lin_v)
        init_attenuated(self.attn.lin_out)

        self.norm_2.reset_parameters()
        self.ffn.reset_parameters()
