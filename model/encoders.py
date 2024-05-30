
from __future__ import annotations

import torch
from torch import Tensor
import torch.nn as nn

from model.embeddings import SurveyEmbeddings

from torch_frame.nn.conv import ExcelFormerConv
from torch_frame.nn.decoder import ExcelFormerDecoder


class CustomExcelFormer(nn.Module):
    r"""The ExcelFormer model introduced in the
    `"ExcelFormer: A Neural Network Surpassing GBDTs on Tabular Data"
    <https://arxiv.org/abs/2301.02819>`_ paper.

    ExcelFormer first converts the categorical features with a target
    statistics encoder (i.e., :class:`CatBoostEncoder` in the paper)
    into numerical features. Then it sorts the numerical features
    with mutual information sort. So the model itself limits to
    numerical features.

    .. note::

        For an example of using ExcelFormer, see `examples/excelformer.py
        <https://github.com/pyg-team/pytorch-frame/blob/master/examples/
        excelfromer.py>`_.

    Args:
        vocab_size (int): Number of possible answers
        n_years (int): Number of years in the dataset
        hidden_size (int): Input channel dimensionality
        out_size (int): Output channels dimensionality
        sequence_len (int): Number of columns
        num_layers (int): Number of
            :class:`torch_frame.nn.conv.ExcelFormerConv` layers.
        num_heads (int): Number of attention heads used in :class:`DiaM`
        diam_dropout (float, optional): diam_dropout. (default: :obj:`0.0`)
        aium_dropout (float, optional): aium_dropout. (default: :obj:`0.0`)
        residual_dropout (float, optional): residual dropout.
            (default: :obj:`0.0`)
    """

    def __init__(
        self,
        vocab_size: int,
        sequence_len: int,
        hidden_size: int,
        out_size: int,
        n_years: int = 14,
        num_layers: int = 3,
        num_heads: int = 4,
        diam_dropout: float = 0.0,
        aium_dropout: float = 0.0,
        embedding_dropout: float = 0.0,
        residual_dropout: float = 0.0,
        layer_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError(
                f"num_layers must be a positive integer (got {num_layers})")

        self.hidden_size = hidden_size
        self.out_size = out_size

        self.embedding = SurveyEmbeddings(vocab_size=vocab_size,
                                          embedding_dim=hidden_size,
                                          n_questions=sequence_len,
                                          n_years=n_years,
                                          dropout=embedding_dropout)

        self.embedding_norm = nn.InstanceNorm1d(sequence_len, affine=False)

        self.excelformer_convs = nn.ModuleList([
            ExcelFormerConv(hidden_size, sequence_len, num_heads, diam_dropout,
                            aium_dropout, residual_dropout)
            for _ in range(num_layers)
        ])
        self.excelformer_decoder = ExcelFormerDecoder(hidden_size,
                                                      out_size, sequence_len)
        self.reset_parameters()
        self.excelformer_decoder.activation = nn.Mish()

        # LayerDrop Placeholder
        self.sampler = torch.distributions.binomial.Binomial(
            probs=1 - layer_dropout)

    def reset_parameters(self) -> None:
        self.embedding.reset_parameters()
        for excelformer_conv in self.excelformer_convs:
            excelformer_conv.reset_parameters()
        self.excelformer_decoder.reset_parameters()

    def forward(self, year: Tensor, seq: Tensor) -> Tensor | tuple[Tensor, Tensor]:
        r"""Transform :class:`TensorFrame` object into output embeddings. If
        `mixup_encoded` is `True`, it produces the output embeddings
        together with the mixed-up targets in `self.mixup` manner.

        Args:

        Returns:
            torch.Tensor | tuple[Tensor, Tensor]: The output embeddings of size
            [batch_size, out_channels]. If `mixup_encoded` is `True`, return
            the mixed-up targets of size [batch_size, num_classes] as well.
        """
        x = self.embedding(year=year, answer=seq)

        for excelformer_conv in self.excelformer_convs:
            x = excelformer_conv(x)

        out = self.excelformer_decoder(x)
        return out, None
