"""
Utility functions for data processing and visualization.
"""

import torch
import numpy as np
import zlib



def compressed_bits_and_rates(s: str) -> tuple[np.ndarray, np.ndarray]:
    """
    For string s of length seq_len, for each i in 1..seq_len:
    - Compress s[:i] with zlib and measure size in bits.
    - Compute compression_rate[i] = compressed_bits / num_chars (ratio; < 1 means compression).

    Returns:
        compressed_bits: shape (seq_len,) - compressed size in bits for s[:1], s[:2], ..., s[:seq_len]
        compression_rates: shape (seq_len,) - compressed_bits[i] / num_chars for index i
    """
    seq_len = len(s)
    compressed_bits = np.zeros(seq_len, dtype=np.float64)
    compression_rates = np.zeros(seq_len, dtype=np.float64)

    for i in range(1, seq_len + 1):
        prefix = s[:i]
        raw_bytes = prefix.encode("utf-8")
        compressed_bytes = zlib.compress(raw_bytes)
        bits = len(compressed_bytes) * 8
        num_chars = len(raw_bytes)
        compressed_bits[i - 1] = bits
        compression_rates[i - 1] = bits / num_chars

    return compressed_bits, compression_rates


def get_delta_compression_rates(s: str) -> np.ndarray:
    """
    For string s of length seq_len, for each i in 1..seq_len:
    - Compute compression_rate[i] = compressed_bits[i] / num_chars
    - Compute delta[i] = compression_rates[i] - compression_rates[i-1]
    """
    seq_len = len(s)
    delta = np.zeros(seq_len - 1, dtype=np.float64)

    prev_num_bits = 0

    for i in range(0, seq_len):
        prefix = s[:i + 1]
        raw_bytes = prefix.encode("utf-8")
        compressed_bytes = zlib.compress(raw_bytes)
        num_bits = len(compressed_bytes) * 8
        if i > 0:
            delta[i - 1] = num_bits - prev_num_bits
        prev_num_bits = num_bits

    return delta



def convert_data(sequences: torch.Tensor, vocab_size: int = 26) -> list[str]:
    """
    Convert sequences of token indices to strings using a character mapping.
    
    For vocab_size=26, maps indices 0-25 to 'a'-'z'.
    For other vocab sizes, uses a generic mapping.
    
    Args:
        sequences: (N, L) or (L,) tensor of token indices
        vocab_size: Size of vocabulary
        
    Returns:
        List of strings, one per sequence (or single string if 1D input)
    """
    # Handle 1D tensor
    if sequences.dim() == 1:
        sequences = sequences.unsqueeze(0)
        single_sequence = True
    else:
        single_sequence = False
    
    # Create character mapping
    if vocab_size <= 26:
        # Map to lowercase letters a-z
        chars = [chr(ord('a') + i) for i in range(26)]
    elif vocab_size <= 52:
        # Map to a-z, then A-Z
        chars = [chr(ord('a') + i) for i in range(26)]
        chars += [chr(ord('A') + i) for i in range(vocab_size - 26)]
    elif vocab_size <= 62:
        # Map to a-z, A-Z, then 0-9
        chars = [chr(ord('a') + i) for i in range(26)]
        chars += [chr(ord('A') + i) for i in range(26)]
        chars += [str(i) for i in range(vocab_size - 52)]
    else:
        # For larger vocabularies, use generic tokens
        chars = [f"<{i}>" for i in range(vocab_size)]
    
    # Convert sequences to strings
    result = []
    for seq in sequences:
        # Convert each token index to its character
        string = ''.join(chars[idx.item()] for idx in seq)
        result.append(string)
    
    return result[0] if single_sequence else result


def convert_sequence_to_string(sequence: torch.Tensor, vocab_size: int = 26) -> str:
    """
    Convert a single sequence to a string.
    Convenience wrapper around convert_data for single sequences.
    
    Args:
        sequence: (L,) tensor of token indices
        vocab_size: Size of vocabulary
        
    Returns:
        String representation of the sequence
    """
    return convert_data(sequence, vocab_size=vocab_size)


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


def generate_random_transition_matrix(vocab_size: int, alpha: torch.Tensor | None = None) -> torch.Tensor:
    """Generate a random row-stochastic Markov transition matrix (vocab_size, vocab_size)."""
    # Sample each row from Dirichlet(alpha) so rows sum to 1
    if alpha is None:
        alpha = torch.ones(vocab_size)
    else:
        assert alpha.size(0) == vocab_size, "alpha must be of size (vocab_size,)"
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
    tilt_factor: float = 0.1,
    seed: int | None = None,
    class_type: str = "same",
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
        tilt_factor: Tilt factor for the transition matrices.
        seed: Random seed for reproducibility.
        class_type: Type of class to generate.
            "same": Transition matrices from the same class.
            "diff": Transition matrices from different classes.

    Returns:
        sequences: (num_samples, seq_len) long tensor of token IDs.
        switch_points: (num_samples,) long tensor of switch indices.
        P0: (vocab_size, vocab_size) first transition matrix.
        P1: (vocab_size, vocab_size) second transition matrix.
    """
    if seed is not None:
        torch.manual_seed(seed)

    if class_type == "same":
        P0 = generate_random_transition_matrix(vocab_size)
        P1 = generate_random_transition_matrix(vocab_size)
    elif class_type == "diff":
        alpha0 = torch.ones(vocab_size)
        alpha1 = torch.ones(vocab_size)
        alpha0[vocab_size // 2:] = tilt_factor
        alpha1[:vocab_size // 2] = tilt_factor
        P0 = generate_random_transition_matrix(vocab_size, alpha0)
        P1 = generate_random_transition_matrix(vocab_size, alpha1)
    else:
        raise ValueError(f"Invalid class type: {class_type}")

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

