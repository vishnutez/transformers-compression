# Evaluate the compression rate of the trained model on the test set
# Computes mean log loss (base 2) = bits per token on the test dataset.
# Distributed: run with torchrun --nproc_per_node=2 eval.py --model_path ...

import argparse
import logging
import math
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import GPT2LMHeadModel

from src.dataset import SwitchingMarkovDataset, generate_transition_matrices
from src.utils import convert_data, compressed_bits_and_rates, get_delta_compression_rates


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
    p.add_argument("--num_chains", type=int, default=10, help="Number of chains for test set")
    p.add_argument("--num_samples_per_chain", type=int, default=100, help="Samples per chain")
    p.add_argument("--window_len", type=int, default=1, help="Window length for switch point")
    p.add_argument("--tilt", type=float, default=0.1, help="Tilt for transition matrices")
    p.add_argument("--vocab_size", type=int, default=26, help="Vocabulary size")
    p.add_argument("--seq_len", type=int, default=1024, help="Sequence length")
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

    # Per-position mean log loss (base 2): average over batch only -> shape (seq_len,)
    seq_len = max_seq_len - 1
    total_loss_bits = torch.zeros(seq_len, device=device, dtype=torch.float64)
    total_samples = 0

    total_zlib_compression_rates = torch.zeros(seq_len, device=device, dtype=torch.float64)
    total_zlib_samples = 0

    total_optimal_loss_bits = torch.zeros(seq_len, device=device, dtype=torch.float64)
    total_optimal_samples = 0

    alphas = torch.ones(2, vocab_size, device=device, dtype=torch.float64)
    alphas[0, vocab_size // 2:] = args.tilt
    alphas[1, :vocab_size // 2] = args.tilt

    with torch.no_grad():
          
        for batch in test_loader:
            input_ids = batch["input_ids"][:, :-1].to(device)
            labels = batch["labels"][:, 1:].to(device)  # (B, L)
            switch_points = batch["switch_t"].to(device)  # (B,)

            
            B = labels.shape[0]
            batch_idx = torch.arange(B, device=device)
            transition_counts = torch.zeros(B, vocab_size, vocab_size, device=device, dtype=torch.long)

            for t in range(seq_len):
                # Reset transition counts at switch points
                transition_counts[t == switch_points] = 0

                # Update counts for all samples in batch
                transition_counts[batch_idx, input_ids[:, t], labels[:, t]] += 1

                # Select prior: alphas[0] before switch, alphas[1] after
                prior = alphas[(t > switch_points).long()]  # (B, vocab_size)

                # Row counts for current input state
                last_state_counts = transition_counts[batch_idx, input_ids[:, t]]  # (B, vocab_size)

                # Posterior
                posterior = (last_state_counts + prior) / (
                    last_state_counts.sum(dim=1, keepdim=True) + prior.sum(dim=1, keepdim=True)
                )  # (B, vocab_size)

                # Log loss in bits, summed across batch
                loss_bits = -posterior[batch_idx, labels[:, t]].log() / math.log(2)  # (B,)
                total_optimal_loss_bits[t] += loss_bits.sum()
                
            total_optimal_samples += labels.shape[0]

    # Reduce across ranks
    if distributed:
        dist.all_reduce(total_optimal_loss_bits, op=dist.ReduceOp.SUM)
        total_optimal_samples_t = torch.tensor([total_optimal_samples], device=device, dtype=torch.long)
        dist.all_reduce(total_optimal_samples_t, op=dist.ReduceOp.SUM)
        total_optimal_samples = total_optimal_samples_t.item()

    mean_optimal_loss_bits = (total_optimal_loss_bits / total_optimal_samples).cpu() if total_optimal_samples > 0 else torch.zeros(seq_len)

    if rank == 0:
        print(f"Test set: {total_optimal_samples} samples (world_size={world_size})", flush=True)
        print(f"mean_optimal_loss_bits shape: {mean_optimal_loss_bits.shape}", flush=True)
        print(f"Mean optimal log loss (base 2) per position (first 5): {mean_optimal_loss_bits[:5].tolist()}", flush=True)
        mean_optimal_loss_bits_np = mean_optimal_loss_bits.numpy()
        np.save(f"metrics/mean_optimal_log_loss_bits_seed={args.seed}_l={seq_len}_w={args.window_len}_nc={args.num_chains}_ns={args.num_samples_per_chain}_t={args.tilt}_v={vocab_size}.npy", mean_optimal_loss_bits_np)
        PURPLE = "#741b47"
        ORANGE = "#ff9100"
        GRAY = "#666666"

        from matplotlib import pyplot as plt

        # Set fontsize and style
        plt.rcParams.update({'font.size': 16})
        plt.rcParams.update({'font.family': 'DejaVu Sans'})

        # Load mean log loss bits from file
        mean_log_loss_bits = np.load(f"metrics/mean_log_loss_bits_seed={args.seed}_l={seq_len}_w={args.window_len}_nc={args.num_chains}_ns={args.num_samples_per_chain}_t={args.tilt}_v={vocab_size}.npy")
        mean_zlib_compression_rates = np.load(f"metrics/mean_zlib_compression_rates_seed={args.seed}_l={seq_len}_w={args.window_len}_nc={args.num_chains}_ns={args.num_samples_per_chain}_t={args.tilt}_v={vocab_size}.npy")
        mean_optimal_loss_bits = np.load(f"metrics/mean_optimal_log_loss_bits_seed={args.seed}_l={seq_len}_w={args.window_len}_nc={args.num_chains}_ns={args.num_samples_per_chain}_t={args.tilt}_v={vocab_size}.npy")

        plt.plot(mean_log_loss_bits, label="transformer", color=PURPLE)
        plt.plot(mean_zlib_compression_rates, label="zlib", color=GRAY, linestyle='--')
        plt.plot(mean_optimal_loss_bits, label="optimal", color=ORANGE)
        plt.xlabel("sequence length")
        plt.ylabel("compression rate (bpc)")
        plt.title(f"window: {args.window_len}")
        plt.grid(True, alpha=0.5)
        plt.legend()
        plt.tight_layout()

        plt.savefig(f"plots/optimal_vs_zlib_vs_transformer_seed={args.seed}_l={max_seq_len}_w={args.window_len}_nc={args.num_chains}_ns={args.num_samples_per_chain}_t={args.tilt}_v={vocab_size}.png")
    
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
