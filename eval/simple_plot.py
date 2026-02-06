# Load mean log loss bits from file
        
from matplotlib import pyplot as plt
import numpy as np

import argparse

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=108)
    p.add_argument("--seq_len", type=int, default=1024)
    p.add_argument("--window_len", type=int, default=1)
    p.add_argument("--num_chains", type=int, default=10)
    p.add_argument("--num_samples_per_chain", type=int, default=50)
    p.add_argument("--tilt", type=float, default=0.1)
    p.add_argument("--vocab_size", type=int, default=26)
    return p.parse_args()

args = parse_args()

seq_len = args.seq_len
vocab_size = args.vocab_size

PURPLE = "#741b47"
ORANGE = "#ff9100"
GRAY = "#666666"

mean_log_loss_bits = np.load(f"metrics/mean_log_loss_bits_seed={args.seed}_l={seq_len}_w={args.window_len}_nc={args.num_chains}_ns={args.num_samples_per_chain}_t={args.tilt}_v={vocab_size}.npy")
mean_zlib_compression_rates = np.load(f"metrics/mean_zlib_compression_rates_seed={args.seed}_l={seq_len}_w={args.window_len}_nc={args.num_chains}_ns={args.num_samples_per_chain}_t={args.tilt}_v={vocab_size}.npy")
mean_optimal_loss_bits = np.load(f"metrics/mean_optimal_log_loss_bits_seed={args.seed}_l={seq_len}_w={args.window_len}_nc={args.num_chains}_ns={args.num_samples_per_chain}_t={args.tilt}_v={vocab_size}.npy")


mid = len(mean_zlib_compression_rates) // 2

zoom_len = 128
start = mid - zoom_len // 2
end = mid + zoom_len // 2

# Set fontsize and style
plt.rcParams.update({'font.size': 16})
plt.rcParams.update({'font.family': 'DejaVu Sans'})

plt.plot(range(start, end), mean_log_loss_bits[start:end], label="Transformer", color=PURPLE)
plt.plot(range(start, end), mean_zlib_compression_rates[start:end], label="Zlib", color=GRAY)
plt.plot(range(start, end), mean_optimal_loss_bits[start:end], label="Bayesian", color=ORANGE)
plt.xlabel("Sequence Length")
plt.ylabel("Compression Rate (bpc)")
plt.title(f"Window: {args.window_len}")
plt.grid(True, alpha=0.5)
plt.legend()
plt.tight_layout()

plt.savefig(f"plots/zoom_len={zoom_len}_seed={args.seed}_l={seq_len}_w={args.window_len}_nc={args.num_chains}_ns={args.num_samples_per_chain}_t={args.tilt}_v={vocab_size}.png")