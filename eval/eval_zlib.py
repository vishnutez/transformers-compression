# Evaluate the compression rate of the trained model on the test set
# Computes mean log loss (base 2) = bits per token on the test dataset.
# Distributed: run with torchrun --nproc_per_node=2 eval.py --model_path ...

import argparse
import logging
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from matplotlib import pyplot as plt

from src.dataset import SwitchingMarkovDataset, generate_transition_matrices
from src.utils import convert_data, compressed_bits_and_rates


def setup_distributed():
    """Initialize process group if RANK/WORLD_SIZE are set (e.g. by torchrun)."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        return True
    return False


def parse_args():
    p = argparse.ArgumentParser(
        description="Compute log loss (base 2) on test set using a saved model."
    )
    p.add_argument("--seed", type=int, default=108, help="Random seed for test data")
    p.add_argument("--batch_size", type=int, default=50, help="Evaluation batch size (per GPU)")
    p.add_argument("--num_chains", type=int, default=100, help="Number of chains for test set")
    p.add_argument("--num_samples_per_chain", type=int, default=100, help="Samples per chain")
    p.add_argument("--window_len", type=int, default=32, help="Window length for switch point")
    p.add_argument("--tilt", type=float, default=0.1, help="Tilt for transition matrices")
    p.add_argument("--vocab_size", type=int, default=26, help="Vocabulary size")
    p.add_argument("--seq_len", type=int, default=256, help="Sequence length")
    return p.parse_args()


def main():
    args = parse_args()
    distributed = setup_distributed()

    if distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device("cuda", local_rank)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        if rank != 0:
            logging.getLogger().setLevel(logging.WARNING)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rank = 0
        world_size = 1

    vocab_size = args.vocab_size
    max_seq_len = args.seq_len + 1

    if args.seed is not None:
        torch.manual_seed(args.seed)

    P = generate_transition_matrices(args.num_chains, vocab_size, args.tilt)
    test_dataset = SwitchingMarkovDataset(
        P, args.num_samples_per_chain, max_seq_len, args.window_len, args.seed
    )

    if distributed:
        test_sampler = DistributedSampler(test_dataset, shuffle=False)
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            sampler=test_sampler,
            shuffle=False,
        )
    else:
        test_loader = DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False
        )

    # Per-position mean compression rate: average over batch only -> shape (seq_len,)
    seq_len = max_seq_len - 1
    total_compression_rates = torch.zeros(max_seq_len, device=device, dtype=torch.float64)
    total_samples = 0

    for batch in test_loader:
        token_ids = batch["input_ids"]
        strings = convert_data(token_ids, vocab_size)

        # Compress using zlib and measure the compression rate
        compression_rates = torch.zeros(len(strings), max_seq_len, device=device, dtype=torch.float64)
        for i, s in enumerate(strings):
            _, crates = compressed_bits_and_rates(s)
            compression_rates[i, :] = torch.tensor(crates, device=device, dtype=torch.float64)

        total_compression_rates += compression_rates.sum(dim=0).double()
        total_samples += len(strings)

    # Reduce across ranks
    if distributed:
        dist.all_reduce(total_compression_rates, op=dist.ReduceOp.SUM)
        total_samples_t = torch.tensor([total_samples], device=device, dtype=torch.long)
        dist.all_reduce(total_samples_t, op=dist.ReduceOp.SUM)
        total_samples = total_samples_t.item()

    mean_compression_rates_zlib = (total_compression_rates / total_samples).cpu() if total_samples > 0 else torch.zeros(seq_len)

    if rank == 0:
        print(f"Test set: {total_samples} samples (world_size={world_size})", flush=True)
        print(f"mean_compression_rates_zlib shape: {mean_compression_rates_zlib.shape}", flush=True)
        print(f"Mean compression rate (zlib) per position (first 5): {mean_compression_rates_zlib[:5].tolist()}", flush=True)
        mean_compression_rates_zlib_np = mean_compression_rates_zlib.numpy()
        np.save(f"mean_compression_rates_zlib_seed={args.seed}_l={args.seq_len}_w={args.window_len}_nc={args.num_chains}_ns={args.num_samples_per_chain}_t={args.tilt}_v={args.vocab_size}.npy", mean_compression_rates_zlib_np)
        
        PURPLE = "#741b47"
        ORANGE = "#ff9100"
        GRAY = "#666666"

        # Set fontsize and style
        plt.rcParams.update({'font.size': 16})
        plt.rcParams.update({'font.family': 'Verdana'})

        plt.plot(mean_compression_rates_zlib_np, label="zlib", color=GRAY, linestyle='--')
        plt.xlabel("sequence length")
        plt.ylabel("compression rate")
        plt.title("compression performance")
        plt.ylim(0, 8)
        plt.grid(True, alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"mean_compression_rates_zlib_seed={args.seed}_seq_len={args.seq_len}_window_len={args.window_len}_num_chains={args.num_chains}_num_samples_per_chain={args.num_samples_per_chain}.png")

    if distributed:
        dist.destroy_process_group()

    

if __name__ == "__main__":
    main()
