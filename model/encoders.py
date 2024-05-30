
from __future__ import annotations

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from model.embeddings import SurveyEmbeddings

import torch_frame
from torch_frame.nn.conv import ExcelFormerConv
from torch_frame.nn.decoder import ExcelFormerDecoder


def feature_mixup(
    x: Tensor,
    y: Tensor,
    num_classes: int,
    beta: float | Tensor = 0.5,
    mixup_type: str | None = None,
    mi_scores: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    r"""Mixup :obj: input numerical feature tensor :obj:`x` by swapping some
    feature elements of two shuffled sample samples. The shuffle rates for
    each row is sampled from the Beta distribution. The target `y` is also
    linearly mixed up.

    Args:
        x (Tensor): The input numerical feature.
        y (Tensor): The target.
        num_classes (int): Number of classes.
        beta (float): The concentration parameter of the Beta distribution.
            (default: :obj:`0.5`)
        mixup_type (str, optional): The mixup methods. No mixup if set to
            :obj:`None`, options `feature` and `hidden` are `FEAT-MIX`
            (mixup at feature dimension) and `HIDDEN-MIX` (mixup at
            hidden dimension) proposed in ExcelFormer paper.
            (default: :obj:`None`)
        mi_scores (Tensor, optional): Mutual information scores only used in
            the mixup weight calculation for `FEAT-MIX`.
            (default: :obj:`None`)

    Returns:
        x_mixedup (Tensor): The mixedup numerical feature.
        y_mixedup (Tensor): Transformed target of size
            :obj:`[batch_size, num_classes]`
    """
    assert num_classes > 0
    assert mixup_type in [None, 'feature', 'hidden']

    beta = torch.tensor(beta, dtype=x.dtype, device=x.device)
    beta_distribution = torch.distributions.beta.Beta(beta, beta)
    shuffle_rates = beta_distribution.sample(torch.Size((len(x), 1)))
    shuffled_idx = torch.randperm(len(x), device=x.device)
    assert x.ndim == 3, """
    FEAT-MIX or HIDDEN-MIX is for encoded numerical features
    of size [batch_size, num_cols, in_channels]."""
    b, f, d = x.shape
    if mixup_type == 'feature':
        assert mi_scores is not None
        mi_scores = mi_scores.to(x.device)
        # Hard mask (feature dimension)
        mixup_mask = torch.rand(torch.Size((b, f)),
                                device=x.device) < shuffle_rates
        # L1 normalized mutual information scores
        norm_mi_scores = mi_scores / mi_scores.sum()
        # Mixup weights
        lam = torch.sum(
            norm_mi_scores.unsqueeze(0) * mixup_mask, dim=1, keepdim=True)
        mixup_mask = mixup_mask.unsqueeze(2)
    elif mixup_type == 'hidden':
        # Hard mask (hidden dimension)
        mixup_mask = torch.rand(torch.Size((b, d)),
                                device=x.device) < shuffle_rates
        mixup_mask = mixup_mask.unsqueeze(1)
        # Mixup weights
        lam = shuffle_rates
    else:
        # No mixup
        mixup_mask = torch.ones_like(x, dtype=torch.bool)
        # Fake mixup weights
        lam = torch.ones_like(shuffle_rates)
    x_mixedup = mixup_mask * x + ~mixup_mask * x[shuffled_idx]

    y_shuffled = y[shuffled_idx]
    if num_classes == 1:
        # Regression task or binary classification
        lam = lam.squeeze(1)
        y_mixedup = lam * y + (1 - lam) * y_shuffled
    else:
        # Classification task
        one_hot_y = F.one_hot(y, num_classes=num_classes)
        one_hot_y_shuffled = F.one_hot(y_shuffled, num_classes=num_classes)
        y_mixedup = (lam * one_hot_y + (1 - lam) * one_hot_y_shuffled)
    return x_mixedup, y_mixedup


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
        mixup (str, optional): mixup type.
            :obj:`None`, :obj:`feature`, or :obj:`hidden`.
            (default: :obj:`None`)
        beta (float, optional): Shape parameter for beta distribution to
                calculate shuffle rate in mixup. Only useful when `mixup` is
                not :obj:`None`. (default: :obj:`0.5`)
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
        num_classes: int = 0.0,
        mixup: str | None = None,
        beta: float = 0.5,
    ) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError(
                f"num_layers must be a positive integer (got {num_layers})")

        assert mixup in [None, 'feature', 'hidden']

        self.hidden_size = hidden_size
        self.out_size = out_size
        self.num_classes = num_classes

        self.embedding = SurveyEmbeddings(vocab_size=vocab_size,
                                          embedding_dim=hidden_size,
                                          n_questions=sequence_len,
                                          n_years=n_years,
                                          dropout=embedding_dropout)

        # self.embedding_norm = nn.InstanceNorm1d(sequence_len, affine=True)

        self.excelformer_convs = nn.ModuleList([
            ExcelFormerConv(hidden_size, sequence_len, num_heads, diam_dropout,
                            aium_dropout, residual_dropout)
            for _ in range(num_layers)
        ])
        self.excelformer_decoder = ExcelFormerDecoder(hidden_size,
                                                      out_size, sequence_len)
        self.reset_parameters()
        self.excelformer_decoder.activation = nn.Mish()
        self.mixup = mixup
        self.beta = beta
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
            tf (:class:`torch_frame.TensorFrame`):
                Input :class:`TensorFrame` object.
            mixup_encoded (bool):
                Whether to mixup on encoded numerical features, i.e.,
                `FEAT-MIX` and `HIDDEN-MIX`.

        Returns:
            torch.Tensor | tuple[Tensor, Tensor]: The output embeddings of size
            [batch_size, out_channels]. If `mixup_encoded` is `True`, return
            the mixed-up targets of size [batch_size, num_classes] as well.
        """
        x = self.embedding(year=year, answer=seq)
        # x = self.embedding_norm(x)

        for excelformer_conv in self.excelformer_convs:
            # if self.training and self.sampler.sample() == 0:
            #    continue
            x = excelformer_conv(x)

        out = self.excelformer_decoder(x)

        return out, None
