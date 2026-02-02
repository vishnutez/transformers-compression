"""
Utility functions for data processing and visualization.
"""

import torch


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
