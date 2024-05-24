import torch.nn as nn
import torch
import torch.nn.functional as F
from model.embeddings import SurveyEmbeddings
from model.layers import CustomTabTransformerConv
from torch_frame.nn.conv import TabTransformerConv, ExcelFormerConv
from torch_frame.nn.decoder import ExcelFormerDecoder


def _generate_predictions(seq, vocab_size: int, p: float = 0.2, pm: float = 0.05, missing_token_id: int = 101):
    """
    Generates sequence with randomly substituted elements and the target array.
    Args:
        seq: the actual sequence
        vocab_size: number of words in the vocabulary
        p: ratio of random rubstituions
        p: ratio for switching existing values to missing value

    Returns:
        seq: updated sequence
        target: as a vector, where
            0 - non-missing original word is intact
            1 - non-missing word was randomly substituted 
            2 - true missing word
            3 - fake missing word (aka, we substituted an existing word with missing token)
    """
    targets = torch.zeros(seq.shape, device=seq.device)
    mask = (seq == missing_token_id)
    targets[mask] = 2
    # random substitutions
    n = seq.numel()
    n_subst = int(n * p)  # number of words to substitute with random
    n_m = int(n * pm)  # random of words to substitute with the "missing token"

    # Indices for substitution
    # Get indices of elements that are not 101
    eligible_indices = (seq != 101).nonzero(as_tuple=False).view(-1)

    # Randomly select a subset of eligible indices for substitution
    if len(eligible_indices) < n_subst:
        n_subst = len(eligible_indices)
    if len(eligible_indices) < n_m:
        n_m = len(eligible_indices)

    indices = eligible_indices[torch.randperm(
        len(eligible_indices), device=seq.device)[:n_subst]]

    indices_m = eligible_indices[torch.randperm(
        len(eligible_indices), device=seq.device)[:n_m]]

    # Substitute elements with random values excluding 101
    random_values = torch.randint(
        1, vocab_size - 1, (n_subst,), device=seq.device)
    # Shift values greater than or equal to 101
    random_values[random_values >= 101] += 1

    _seq = seq.view(-1)
    _seq[indices] = random_values

    # Fill X with 3s at the substituted positions
    _targets = targets.view(-1)
    _targets[indices] = 1

    # Substitute elements with 101
    _seq[indices_m] = missing_token_id

    # Set target elements to 4 where substitutions with 101 occurred
    _targets[indices_m] = 3

    # Reshape SEQ and TARGETSS back to its original shape
    targets = _targets.view(targets.shape)
    seq = _seq.view(seq.shape)
    return seq, targets


class TabularEncoder(nn.Module):
    """
    Encoder that models the interaction between the columns of the SurveyEmbeddings
    """

    def __init__(self, vocab_size,
                 sequence_len: int,
                 embedding_size: int,
                 output_size: int,
                 num_layers: int = 3,
                 num_heads: int = 4,
                 dropout: float = 0.1,
                 decoder_output: int = 1,
                 layer_type: str = "attn") -> None:
        """
        Args:
            layer_type: either use convolution of attention to model the interactions
        """
        super().__init__()
        assert layer_type in ["excel", "attn", "mixture"], "Wrong layer type"
        self.num_cols = sequence_len
        self.embedding = SurveyEmbeddings(
            vocab_size=vocab_size, n_questions=self.num_cols, n_years=14, embedding_dim=embedding_size)

        if layer_type == "attn":
            self.encoders = nn.ModuleList([
                nn.Sequential(
                    TabTransformerConv(channels=embedding_size,
                                       num_heads=num_heads,
                                       attn_dropout=dropout,
                                       ffn_dropout=dropout),
                    nn.Mish(),
                    nn.InstanceNorm1d(sequence_len))
                for _ in range(num_layers)])
        elif layer_type == "excel":
            self.encoders = nn.ModuleList([
                nn.Sequential(
                    ExcelFormerConv(channels=embedding_size,
                                    num_cols=self.num_cols,
                                    num_heads=num_heads,
                                    diam_dropout=dropout,
                                    aium_dropout=dropout,
                                    residual_dropout=dropout),
                    nn.Mish(),
                    nn.InstanceNorm1d(sequence_len)
                )
                for _ in range(num_layers)])
        elif layer_type == "mixture":
            self.encoders = nn.ModuleList([
                nn.Sequential(
                    CustomTabTransformerConv(channels=embedding_size,
                                             num_heads=num_heads,
                                             attn_dropout=dropout,
                                             auim_dropout=dropout,
                                             residual_dropout=0.1),
                    nn.Dropout(dropout),
                    nn.Mish(),
                    nn.InstanceNorm1d(sequence_len))
                for _ in range(num_layers)])

        self.decoder = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embedding_size, decoder_output),
        )
        # this layer just aggregates the representation of column into one embedding
        self.flatten = ExcelFormerDecoder(in_channels=embedding_size,
                                          out_channels=output_size,
                                          num_cols=self.num_cols)
        self.flatten.activation = nn.Mish()

    def forward(self, year, seq):
        """
        Method that returns full encoding-decoding
        """
        assert seq.size(1) == self.num_cols, "Wrong shapes"
        x = self.embedding(year, seq)
        for encoder in self.encoders:
            x = encoder(x)
        x = self.decoder(x)
        return x

    def get_embedding(self, year, seq):
        """
        Returns initial embeddings of the sequence
        """
        return self.embedding(year, seq)

    def get_encoding(self, year, seq):
        """
        Method that return the embedding of the survey
        """
        x = self.embedding(year, seq)
        for encoder in self.encoders:
            x = encoder(x)
        x = self.flatten(x)
        return x


class SimpleAutoEncoder(nn.Module):
    def __init__(self, vocab_size, sequence_len: int, embedding_size: int, dropout=0.1) -> None:
        super().__init__()

        self.dropout = dropout

        self.embedding = SurveyEmbeddings(
            vocab_size, sequence_len, n_years=14, embedding_dim=embedding_size)

        self.out = nn.Sequential(
            nn.Linear(embedding_size, vocab_size, bias=False)
        )

        self.encoder = nn.Sequential(
            nn.Linear(embedding_size, embedding_size // 2),
            nn.Mish(),
            nn.LayerNorm(embedding_size // 2),
            nn.Dropout(self.dropout),
            nn.Linear(embedding_size // 2, embedding_size // 4),
            nn.Mish(),
            nn.LayerNorm(embedding_size // 4),
            nn.Dropout(self.dropout),
            nn.Linear(embedding_size // 4, embedding_size // 8),
            nn.LayerNorm(embedding_size // 8),
            nn.Dropout(self.dropout),
        )

        self.decoder = nn.Sequential(
            nn.Linear(embedding_size // 8, embedding_size // 4),
            nn.Mish(),
            nn.LayerNorm(embedding_size // 4),
            nn.Dropout(self.dropout),
            nn.Linear(embedding_size // 4, embedding_size // 2),
            nn.Mish(),
            nn.LayerNorm(embedding_size // 2),
            nn.Dropout(self.dropout),
            nn.Linear(embedding_size // 2, embedding_size),
        )

    def forward(self, year, seq):
        """
        Method that returns full encoding-decoding
        """
        embeddings = self.embedding(year, seq)
        x = self.encoder(embeddings)
        x = self.decoder(x)
        x = self.out(x)
        return x

    def get_encoding(self, year, seq):
        """
        Method that return the embedding of the survey
        """
        embeddings = self.embedding(year, seq)
        x = self.encoder(embeddings)
        return x
