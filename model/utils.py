import torch
import random


def get_device():
    # Check if CUDA is available
    if torch.cuda.is_available():
        # If CUDA is available, select the first CUDA device
        device = torch.device("cuda:0")
        print("Using CUDA device:", torch.cuda.get_device_name(0))
    # Check for MPS availability on supported macOS devices (requires PyTorch 1.12 or newer)
    elif torch.backends.mps.is_available():
        # If MPS is available, use MPS device
        device = torch.device("mps")
        print("Using MPS (Metal Performance Shaders) device")
    else:
        # Fallback to CPU if neither CUDA nor MPS is available
        device = torch.device("cpu")
        print("Using CPU")
    return device


def generate_predictions(seq, vocab_size: int, p: float = 0.2, missing_token_id: int = 101):
    """
    Generates sequence with randomly substituted elements and the target array.
    Args:
        seq: the actual sequence
        vocab_size: number of words in the vocabulary
        p: ratio of random rubstitutions

    Returns:
        seq: updated sequence
        target: as a vector, where
            0 - non-missing original word is intact
            1 - non-missing word was randomly substituted 
            2 - true missing word
    """
    targets = torch.zeros(seq.shape, device=seq.device)
    mask = (seq == missing_token_id)
    targets[mask] = 2
    # random substitutions
    n = seq.numel()
    n_subst = int(n * p)
    # Indices for substitution
    indices = torch.randperm(n, device=seq.device)[: n_subst]

    # Substitute elements with random values
    random_values = torch.randint(1, vocab_size, (n_subst,), device=seq.device)
    _seq = seq.view(-1)
    _seq[indices] = random_values

    # Fill X with 3s at the substituted positions
    _targets = targets.view(-1)
    _targets[indices] = 1

    # Reshape X back to its original shape
    targets = _targets.view(targets.shape)
    seq = _seq.view(seq.shape)
    return seq, targets


def generate_contrastive_task(x: torch.Tensor, swap_probability: float = 0.5):
    """
    Augment the x matrix by randomly swipping some rows (for contrastive task)
    Returns the augmented data, where targets are 
        1 if the row was not swapped
        -1 if the row was swapped


    Parameters:
    x (torch.Tensor): The input data tensor.
    swap_probability (float): The probability of swapping each row. Default is 0.5.

    Returns:
    x (torch.Tensor): The data tensor after swapping.
    targets (torch.Tensor): The target tensor indicating whether a row was swapped (-1) or not (1).
    """
    batch_size = x.size(0)
    targets = torch.ones(batch_size).to(
        x.device)  # Initialize targets with ones

    for i in range(batch_size):
        if random.random() < swap_probability:
            swap_idx = random.randint(0, batch_size - 1)
            if swap_idx != i:
                # Swap the rows in the original data
                x[i], x[swap_idx] = x[swap_idx], x[i]
                # Set the target to -1 for swapped rows
                targets[i] = -1
                targets[swap_idx] = -1

    return x, targets


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
