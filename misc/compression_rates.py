"""
Compute zlib compression rates for prefixes of sequences.

Given a string s of length seq_len, for each i in 1..seq_len we compress
running_str = s[:i] with zlib, measure the number of bits, and build an
array of compression_rates (compressed_bits / original_bits).
"""

import sys
from pathlib import Path

import numpy as np
import zlib
import matplotlib.pyplot as plt

# Add parent directory to path to import src modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import generate_switching_markov_sequences
from src.utils import convert_data


def compressed_bits_and_rates(s: str) -> tuple[np.ndarray, np.ndarray]:
    """
    For string s of length seq_len, for each i in 1..seq_len:
    - Compress s[:i] with zlib and measure size in bits.
    - Compute compression_rate[i] = compressed_bits / (i * 8) (ratio; < 1 means compression).

    Returns:
        compressed_bits: shape (seq_len,) - compressed size in bits for s[:1], s[:2], ..., s[:seq_len]
        compression_rates: shape (seq_len,) - compressed_bits[i] / num_chars for index i
    """
    seq_len = len(s)
    compressed_bits = np.zeros(seq_len, dtype=np.float64)
    compression_rates = np.zeros(seq_len, dtype=np.float64)

    for num_chars in range(1, seq_len + 1):
        prefix = s[:num_chars]
        raw_bytes = prefix.encode("utf-8")
        compressed_bytes = zlib.compress(raw_bytes)
        bits = len(compressed_bytes) * 8
        compressed_bits[num_chars - 1] = bits
        compression_rates[num_chars - 1] = bits / num_chars

    return compressed_bits, compression_rates


def main():
    # Configuration
    num_samples = 1000
    seq_len = 1024
    vocab_size = 26
    window_len = 50
    seed = 42

    print("=" * 60)
    print("Zlib compression rates for sequence prefixes")
    print("=" * 60)
    print(f"Configuration: num_samples={num_samples}, seq_len={seq_len}, vocab_size={vocab_size}")
    print()

    # 1. Generate sequences (token-ID tensors)
    print("Generating sequences...")
    sequences, switch_points, P0, P1 = generate_switching_markov_sequences(
        num_samples=num_samples,
        seq_len=seq_len,
        vocab_size=vocab_size,
        window_len=window_len,
        seed=seed,
    )
    # sequences: (num_samples, seq_len)

    # 2. Convert to data values (strings)
    print("Converting to strings...")
    strings = convert_data(sequences, vocab_size=vocab_size)
    # strings: list of num_samples strings, each of length seq_len

    # 3. For each sequence, compute compressed_bits and compression_rates per prefix
    print("Computing compressed bits and compression rates for each prefix length...")
    compressed_bits = []
    compression_rates = []

    for idx, s in enumerate(strings):
        cbits, crates = compressed_bits_and_rates(s)
        compressed_bits.append(cbits)
        compression_rates.append(crates)

    # Stack into arrays: (num_samples, seq_len)
    compressed_bits = np.stack(compressed_bits, axis=0)
    compression_rates = np.stack(compression_rates, axis=0)

    print()
    print("Results (first sample):")
    print(f"  compressed_bits shape: {compressed_bits.shape}")
    print(f"  compression_rates shape: {compression_rates.shape}")
    print(f"  For sample 0, first 10 prefix lengths:")
    print("    i   compressed_bits   compression_rate")
    for i in range(min(10, seq_len)):
        print(f"    {i+1:3d}   {compressed_bits[0, i]:14.0f}   {compression_rates[0, i]:.4f}")
    print(f"  ...")
    print(f"    {seq_len:3d}   {compressed_bits[0, seq_len-1]:14.0f}   {compression_rates[0, seq_len-1]:.4f}")

    # Summary across samples
    mean_rates = compression_rates.mean(axis=0)
    print()
    print("Mean compression rate across samples (by prefix length):")
    print(f"  At i=1:    {mean_rates[0]:.4f}")
    print(f"  At i=32:   {mean_rates[31]:.4f}")
    print(f"  At i=128:  {mean_rates[127]:.4f}")
    print(f"  At i=seq_len: {mean_rates[seq_len-1]:.4f}")

    # Plot the compression rates and save the plot
    plt.plot(mean_rates, c="k")
    plt.xlabel("length of the sequence")
    plt.ylabel("compression rate (bits per character)")
    plt.title("zlib: switching markov sequences")
    plt.savefig("zlib_compression_rates_switching_markov.png")

    return compressed_bits, compression_rates


if __name__ == "__main__":
    main()
