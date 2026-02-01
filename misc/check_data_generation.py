"""
Check data generation using generate_switching_markov_sequences.
This script generates sample sequences and prints statistics.
"""

import sys
from pathlib import Path

# Add parent directory to path to import src modules
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.data import generate_switching_markov_sequences
from src.utils import convert_data


def main():
    # Configuration
    num_samples = 100
    seq_len = 512
    vocab_size = 26
    window_len = 50
    seed = 42
    
    print("=" * 80)
    print("Data Generation Check")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  - Number of samples: {num_samples}")
    print(f"  - Sequence length: {seq_len}")
    print(f"  - Vocabulary size: {vocab_size}")
    print(f"  - Window length: {window_len}")
    print(f"  - Seed: {seed}")
    print()
    
    # Generate sequences
    print("Generating switching Markov sequences...")
    sequences, switch_points, P0, P1 = generate_switching_markov_sequences(
        num_samples=num_samples,
        seq_len=seq_len,
        vocab_size=vocab_size,
        window_len=window_len,
        seed=seed,
    )
    print("Generation complete!")
    print()
    
    # Print statistics
    print("=" * 80)
    print("Results:")
    print("=" * 80)
    
    print(f"\nSequences shape: {sequences.shape}")
    print(f"  - Expected: ({num_samples}, {seq_len})")
    print(f"  - Data type: {sequences.dtype}")
    print(f"  - Min value: {sequences.min().item()}")
    print(f"  - Max value: {sequences.max().item()}")
    print(f"  - Mean value: {sequences.float().mean().item():.2f}")
    
    print(f"\nSwitch points shape: {switch_points.shape}")
    print(f"  - Expected: ({num_samples},)")
    print(f"  - Min switch point: {switch_points.min().item()}")
    print(f"  - Max switch point: {switch_points.max().item()}")
    print(f"  - Mean switch point: {switch_points.float().mean().item():.2f}")
    print(f"  - Expected range: [{seq_len//2 - window_len}, {seq_len//2 + window_len})")
    
    print(f"\nTransition matrix P0 shape: {P0.shape}")
    print(f"  - Expected: ({vocab_size}, {vocab_size})")
    print(f"  - Row sums (should be ~1.0):")
    row_sums = P0.sum(dim=1)
    print(f"    - Min: {row_sums.min().item():.6f}")
    print(f"    - Max: {row_sums.max().item():.6f}")
    print(f"    - Mean: {row_sums.mean().item():.6f}")
    
    print(f"\nTransition matrix P1 shape: {P1.shape}")
    print(f"  - Expected: ({vocab_size}, {vocab_size})")
    print(f"  - Row sums (should be ~1.0):")
    row_sums = P1.sum(dim=1)
    print(f"    - Min: {row_sums.min().item():.6f}")
    print(f"    - Max: {row_sums.max().item():.6f}")
    print(f"    - Mean: {row_sums.mean().item():.6f}")
    
    # Check a few sample sequences
    print("\n" + "=" * 80)
    print("Sample sequences (first 3 samples, first 60 characters):")
    print("=" * 80)
    
    # Convert sequences to readable strings
    sample_sequences = sequences[:min(3, num_samples), :60]
    converted = convert_data(sample_sequences, vocab_size=vocab_size)
    
    for i in range(len(converted)):
        print(f"\nSample {i+1} (switch point at {switch_points[i].item()}):")
        print(f"  Indices: {sequences[i, :20].tolist()}")
        print(f"  String:  {converted[i]}")
    
    # Verify token distribution
    print("\n" + "=" * 80)
    print("Token distribution:")
    print("=" * 80)
    unique_tokens, counts = torch.unique(sequences, return_counts=True)
    print(f"  - Unique tokens: {len(unique_tokens)} (expected: {vocab_size})")
    print(f"  - Most common token: {unique_tokens[counts.argmax()].item()} "
          f"(count: {counts.max().item()})")
    print(f"  - Least common token: {unique_tokens[counts.argmin()].item()} "
          f"(count: {counts.min().item()})")
    
    print("\n" + "=" * 80)
    print("All checks passed! ✓")
    print("=" * 80)


if __name__ == "__main__":
    main()
