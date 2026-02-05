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
    p.add_argument("--model_path", type=str, required=True, help="Directory with saved model (config + weights)")
    p.add_argument("--seed", type=int, default=108, help="Random seed for test data")
    p.add_argument("--batch_size", type=int, default=50, help="Evaluation batch size (per GPU)")
    p.add_argument("--num_chains", type=int, default=100, help="Number of chains for test set")
    p.add_argument("--num_samples_per_chain", type=int, default=100, help="Samples per chain")
    p.add_argument("--window_len", type=int, default=32, help="Window length for switch point")
    p.add_argument("--tilt", type=float, default=0.1, help="Tilt for transition matrices")
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

    # Load model from checkpoint (config + weights)
    if rank == 0:
        print(f"Loading model from {args.model_path} ...", flush=True)
    model = GPT2LMHeadModel.from_pretrained(args.model_path, local_files_only=True)
    model = model.to(device)
    model.eval()

    # Match dataset to training: vocab_size and seq_len from model config
    vocab_size = model.config.vocab_size
    max_seq_len = model.config.n_positions

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

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"][:, :-1].to(device)
            labels = batch["labels"][:, 1:].to(device)  # (B, L)

            logits = model(input_ids).logits  # (B, L, vocab_size)
            loss_nats = F.cross_entropy(
                logits.view(-1, vocab_size),
                labels.view(-1),
                reduction="none",
                ignore_index=-100,
            )
            loss_nats = loss_nats.view(labels.shape[0], -1)  # (B, L)
            loss_bits = loss_nats / math.log(2)
            total_loss_bits += loss_bits.sum(dim=0).double()
            total_samples += labels.shape[0]

    # Reduce across ranks
    if distributed:
        dist.all_reduce(total_loss_bits, op=dist.ReduceOp.SUM)
        total_samples_t = torch.tensor([total_samples], device=device, dtype=torch.long)
        dist.all_reduce(total_samples_t, op=dist.ReduceOp.SUM)
        total_samples = total_samples_t.item()

    mean_log_loss_bits = (total_loss_bits / total_samples).cpu() if total_samples > 0 else torch.zeros(seq_len)

    if rank == 0:
        print(f"Test set: {total_samples} samples (world_size={world_size})", flush=True)
        print(f"mean_log_loss_bits shape: {mean_log_loss_bits.shape}", flush=True)
        print(f"Mean log loss (base 2) per position (first 5): {mean_log_loss_bits[:5].tolist()}", flush=True)
        mean_log_loss_bits_np = mean_log_loss_bits.numpy()
        np.save(f"mean_log_loss_bits_seed={args.seed}_l={seq_len}_w={args.window_len}_nc={args.num_chains}_ns={args.num_samples_per_chain}_t={args.tilt}_v={vocab_size}.npy", mean_log_loss_bits_np)

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
