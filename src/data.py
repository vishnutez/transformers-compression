"""
Data handling for finite-vocabulary sequences (token IDs only).
No tokenizer: vocab is integers in [0, vocab_size).
"""

import torch
from torch.utils.data import Dataset

 
def generate_random_sequences(
    num_sequences: int,
    seq_len: int,
    vocab_size: int,
    seed: int | None = None,
) -> torch.Tensor:
    """Generate random token-ID sequences for debugging or synthetic data."""
    if seed is not None:
        torch.manual_seed(seed)
    return torch.randint(0, vocab_size, (num_sequences, seq_len))


def generate_random_transition_matrix(vocab_size: int) -> torch.Tensor:
    """Generate a random row-stochastic Markov transition matrix (vocab_size, vocab_size)."""
    # Sample each row from Dirichlet(alpha) so rows sum to 1
    alpha = torch.ones(vocab_size)
    dist = torch.distributions.Dirichlet(alpha)
    return dist.sample((vocab_size,))


def generate_markov_sequence(
    seq_len: int,
    transition_probs: torch.Tensor,
    initial_state: int | None = None,
) -> torch.Tensor:
    """
    Generate a single Markov chain sequence of length seq_len.

    Args:
        seq_len: Length of the sequence.
        transition_probs: (vocab_size, vocab_size) row-stochastic transition matrix.
        initial_state: Starting state. If None, drawn uniformly.

    Returns:
        (seq_len,) long tensor of token IDs.
    """
    vocab_size = transition_probs.size(0)
    seq = torch.empty(seq_len, dtype=torch.long)

    if initial_state is None:
        state = torch.randint(0, vocab_size, ()).item()
    else:
        state = initial_state

    for t in range(seq_len):
        seq[t] = state
        # Sample next state from transition_probs[state]
        state = torch.multinomial(transition_probs[state], num_samples=1).item()

    return seq


def generate_switching_markov_sequences(
    num_samples: int,
    seq_len: int,
    vocab_size: int,
    window_len: int,
    seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate sequences from a Markov chain that switches between two transition matrices.

    For each sequence:
    - Draw switch_point uniformly from (mid - window_len, mid + window_len)
    - For positions [0, switch_point): use P0
    - For positions [switch_point, seq_len): use P1

    The two transition matrices are shared across all samples; only switch_point
    and the actual sequences differ per sample.

    Args:
        num_samples: Number of sequences to generate.
        seq_len: Length of each sequence.
        vocab_size: Size of vocabulary / number of Markov states.
        window_len: Half-width of the window around mid for the switch point.
        seed: Random seed for reproducibility.

    Returns:
        sequences: (num_samples, seq_len) long tensor of token IDs.
        switch_points: (num_samples,) long tensor of switch indices.
        P0: (vocab_size, vocab_size) first transition matrix.
        P1: (vocab_size, vocab_size) second transition matrix.
    """
    if seed is not None:
        torch.manual_seed(seed)

    # Generate shared transition matrices
    P0 = generate_random_transition_matrix(vocab_size)
    P1 = generate_random_transition_matrix(vocab_size)

    mid = seq_len // 2
    low = max(0, mid - window_len)
    high = min(seq_len, mid + window_len)

    sequences = torch.empty(num_samples, seq_len, dtype=torch.long)
    switch_points = torch.randint(low, high, (num_samples,))

    for i in range(num_samples):
        switch_pt = switch_points[i].item()

        # Generate first segment [0, switch_pt) using P0
        if switch_pt > 0:
            seg0 = generate_markov_sequence(switch_pt, P0)
            sequences[i, :switch_pt] = seg0
            last_state = seg0[-1].item()
        else:
            last_state = torch.randint(0, vocab_size, ()).item()

        # Generate second segment [switch_pt, seq_len) using P1
        if switch_pt < seq_len:
            seg1 = generate_markov_sequence(seq_len - switch_pt, P1, initial_state=last_state)
            sequences[i, switch_pt:] = seg1

    return sequences, switch_points, P0, P1


class FiniteVocabDataset(Dataset):
    """
    Dataset over sequences of token IDs in [0, vocab_size).
    Each sample is (input_ids, target_ids) for next-token prediction:
    target_ids[t] = input_ids[t+1].
    """

    def __init__(self, sequences: torch.Tensor, vocab_size: int):
        """
        Args:
            sequences: (N, L) long tensor of token IDs.
            vocab_size: size of vocabulary (ids must be in [0, vocab_size)).
        """
        assert sequences.dtype in (torch.long, torch.int64)
        assert sequences.dim() == 2
        self.sequences = sequences
        self.vocab_size = vocab_size

    def __len__(self) -> int:
        return self.sequences.size(0)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        seq = self.sequences[idx].clone()  # (L,)
        # NTP: predict next token at each position. HF causal LM expects
        # input_ids + labels (model shifts internally).
        return {"input_ids": seq, "labels": seq}
