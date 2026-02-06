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

            
            for b in range(len(labels)):
                # Create a transition count matrix
                transition_counts = torch.zeros(vocab_size, vocab_size, device=device, dtype=torch.long)
                for t in range(seq_len):
                    # Reset transition counts to zero at switch point
                    if t == switch_points[b].item(): # switch point
                        transition_counts = torch.zeros(vocab_size, vocab_size, device=device, dtype=torch.long)

                    transition_counts[input_ids[b, t], labels[b, t]] += 1
                    prior = alphas[t > switch_points[b]]  # (vocab_size,)
                    last_state_counts = transition_counts[input_ids[b, t]] # (vocab_size,)
                    posterior = (last_state_counts + prior) / (last_state_counts.sum() + prior.sum())  # (vocab_size,)

                    next_state = labels[b, t].to(device)
                    loss_nats = -posterior[next_state].log().item()
                    loss_bits = loss_nats.item() / math.log(2)
                    total_optimal_loss_bits[t] += loss_bits
                
            total_optimal_samples += labels.shape[0]

            # logits = model(input_ids).logits  # (B, L, vocab_size)
            # loss_nats = F.cross_entropy(
            #     logits.view(-1, vocab_size),
            #     labels.view(-1),
            #     reduction="none",
            #     ignore_index=-100,
            # )
            # loss_nats = loss_nats.view(labels.shape[0], -1)  # (B, L)
            # loss_bits = loss_nats / math.log(2)
            # total_loss_bits += loss_bits.sum(dim=0).double()
            # total_samples += labels.shape[0]

            # token_ids = batch["input_ids"].to(device) # (B, L + 1) for zlib
            # strings = convert_data(token_ids, vocab_size)

            # # compression_rates = torch.zeros(len(strings), max_seq_len, device=device, dtype=torch.float64)
            # # for i, s in enumerate(strings):
            # #     _, crates = compressed_bits_and_rates(s)
            # #     compression_rates[i, :] = torch.tensor(crates, device=device, dtype=torch.float64)
            # # total_zlib_compression_rates += compression_rates.sum(dim=0).double()

            # delta_compression_rates = torch.zeros(len(strings), seq_len, device=device, dtype=torch.float64)
            # for i, s in enumerate(strings):
            #     delta_rates = get_delta_compression_rates(s)
            #     delta_compression_rates[i, :] = torch.tensor(delta_rates, device=device, dtype=torch.float64)
            # total_zlib_compression_rates += delta_compression_rates.sum(dim=0).double()
            # total_zlib_samples += len(strings)

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

        # Set fontsize and style
        plt.rcParams.update({'font.size': 16})
        plt.rcParams.update({'font.family': 'DejaVu Sans'})

        # Load mean log loss bits from file
        mean_log_loss_bits = np.load(f"metrics/mean_log_loss_bits_seed={args.seed}_l={max_seq_len}_w={args.window_len}_nc={args.num_chains}_ns={args.num_samples_per_chain}_t={args.tilt}_v={vocab_size}.npy")
        mean_zlib_compression_rates = np.load(f"metrics/mean_zlib_compression_rates_seed={args.seed}_l={max_seq_len}_w={args.window_len}_nc={args.num_chains}_ns={args.num_samples_per_chain}_t={args.tilt}_v={vocab_size}.npy")
        mean_optimal_loss_bits = np.load(f"metrics/mean_optimal_log_loss_bits_seed={args.seed}_l={max_seq_len}_w={args.window_len}_nc={args.num_chains}_ns={args.num_samples_per_chain}_t={args.tilt}_v={vocab_size}.npy")

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
