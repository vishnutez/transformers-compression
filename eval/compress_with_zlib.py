import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import (compressed_bits_and_rates, generate_markov_sequence,
                       convert_data, generate_random_transition_matrix, convert_sequence_to_string)
import zlib
import torch
from matplotlib import pyplot as plt

seq_len = 50_000
tilt = 0.1

tilt_t = torch.tensor(tilt)

entropy_rate = (-tilt_t * torch.log2(tilt_t) - (1 - tilt_t) * torch.log2(1 - tilt_t)).item()
print(f"Entropy rate: {entropy_rate}")


# vocab_size = 26
# alpha = torch.ones(vocab_size)
# alpha[vocab_size // 2:] = tilt
# P = generate_random_transition_matrix(vocab_size, alpha)

# Test case: 2-state Markov chain with equal probabilities
P = torch.tensor([[tilt, 1 - tilt], [1 - tilt, tilt]])


print("Generating sequence...", flush=True)
sequence = generate_markov_sequence(seq_len, P)
print("Converting sequence to string...", flush=True)
string = convert_sequence_to_string(sequence)
print("Sequence generated and converted to string", flush=True)


# Compress with zlib

num_chars = len(string)
spacing = 100
num_chunks = num_chars // spacing

crates = torch.zeros(num_chunks)
for i in range(num_chunks):
    chunk = string[:(i + 1) * spacing]
    raw_bytes = chunk.encode('utf-8')
    compressed = zlib.compress(raw_bytes, level=9)
    crates[i] = len(compressed) * 8.0 / len(raw_bytes)
    print(f"chunk {i} of length {len(chunk)} has compression ratio: {crates[i]}", flush=True)


PURPLE = "#741b47"
GRAY = "#666666"
ORANGE = "#ff9100"

# Set fontsize and style
plt.rcParams.update({'font.size': 16})
plt.rcParams.update({'font.family': 'DejaVu Sans'})


plt.semilogx(range(0, num_chars, spacing), crates, label="zlib", linestyle= '--', color=GRAY)
plt.axhline(entropy_rate, color=ORANGE, label="entropy rate")
plt.xlabel("number of characters")
plt.ylabel("compression ratio")
plt.title("zlib on long sequence")
plt.legend()
plt.tight_layout()
plt.savefig("binary_zlib_long_sequence_compression_ratio.png")