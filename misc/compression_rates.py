"""
Compute zlib compression rates for prefixes of sequences.

Given a string s of length seq_len, for each i in 1..seq_len we compress
running_str = s[:i] with zlib, measure the number of bits, and build an
array of compression_rates (compressed_bits / num_chars).
"""

import sys
from pathlib import Path

import numpy as np
import zlib
import matplotlib.pyplot as plt
import torch

# Add parent directory to path to import src modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import generate_switching_markov_sequences, generate_markov_sequence
from src.utils import convert_data


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


def main():
    # Configuration
    num_samples = 500
    seq_len = 128
    vocab_size = 26
    window_len = 32
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    mid = seq_len // 2
    class_type = "same"
    tilt_factor = 1.0

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
        class_type=class_type,
        tilt_factor=tilt_factor,
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
    for i in range(seq_len - 10, seq_len):
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

    # Compute compression rates for sequences without switching
    sequences_wo_switching = sequences.clone()
    for i, switch_pt in enumerate(switch_points):
        last_state = sequences[i, switch_pt - 1]
        sequences_wo_switching[i, switch_pt:] = generate_markov_sequence(seq_len - switch_pt, P0, last_state)
    strings_wo_switching = convert_data(sequences_wo_switching, vocab_size=vocab_size)
    compressed_bits_wo_switching = []
    compression_rates_wo_switching = []
    for s in strings_wo_switching:
        cbits, crates = compressed_bits_and_rates(s)
        compressed_bits_wo_switching.append(cbits)
        compression_rates_wo_switching.append(crates)
    compressed_bits_wo_switching = np.stack(compressed_bits_wo_switching, axis=0)
    compression_rates_wo_switching = np.stack(compression_rates_wo_switching, axis=0)

    mean_rates_wo_switching = compression_rates_wo_switching.mean(axis=0)  # w/o switching < w/ switching

    print()
    print("Results (first sample) for sequences without switching:")
    print(f"  compressed_bits shape: {compressed_bits_wo_switching.shape}")
    print(f"  compression_rates shape: {compression_rates_wo_switching.shape}")
    print(f"  For sample 0, first 10 prefix lengths:")
    print("    i   compressed_bits   compression_rate")
    for i in range(seq_len - 10, seq_len):
        print(f"    {i+1:3d}   {compressed_bits_wo_switching[0, i]:14.0f}   {compression_rates_wo_switching[0, i]:.4f}")
    print(f"    ...")
    print(f"    {seq_len:3d}   {compressed_bits_wo_switching[0, seq_len-1]:14.0f}   {compression_rates_wo_switching[0, seq_len-1]:.4f}")


    PURPLE = "#741b47"
    ORANGE = "#ff9100"
    GRAY = "#666666"

    font_size = 16
    plt.rcParams.update({'font.size': font_size})
    plt.rcParams.update({'font.family': 'Verdana'})

    # Plot the compression rates and save the plot
    plt.plot(mean_rates, c="k", label="w/ switching", linewidth=1)
    plt.plot(mean_rates_wo_switching, c=PURPLE, label="w/o switching", linewidth=3, linestyle="--")
    plt.axhline(y=8, color=ORANGE, linestyle="dotted", label='1 byte = 8 bits')
    plt.axvline(x=mid, color=GRAY, linestyle="dotted", label=r'$\frac{N}{2}$')
    plt.xlabel(r"length of the sequence ($N$)")
    plt.ylabel(r"compression rate (bpc)")
    plt.title(f"zlib: {class_type} classes")
    plt.ylim(0, 8)
    plt.grid(True, linestyle="-", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"zlib_compression_rates_{class_type}_classes_{seq_len}_tilt_{tilt_factor}_n{num_samples}_seed{seed}.png")

    plt.close()

    loss_in_rate = mean_rates - mean_rates_wo_switching  # rate w/ switching > rate w/o switching => loss >= 0
    print(f"Difference between mean rates: {loss_in_rate}")

    plt.plot(loss_in_rate, c=PURPLE, linewidth=2)
    plt.xlabel(r"length of the sequence ($N$)")
    plt.ylabel(r"loss in rate (bpc)")
    plt.title(f"compression rates: {class_type} classes")
    plt.grid(True, linestyle="-", alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"zlib_loss_in_rate_{class_type}_classes_{seq_len}_tilt_{tilt_factor}_n{num_samples}_seed{seed}.png")

if __name__ == "__main__":
    main()
