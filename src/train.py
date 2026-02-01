"""
Train the NTP model (Hugging Face GPT-2) on finite-vocab sequences with next-token prediction loss.
Uses Hugging Face Trainer for distributed training. For 2 GPUs on one node, run:

    accelerate launch --num_processes 2 train.py [args]

or:

    torchrun --nproc_per_node=2 train.py [args]
"""

import argparse
import os

import torch
import wandb
from transformers import Trainer, TrainingArguments

from data import FiniteVocabDataset, generate_random_sequences
from models import create_ntp_model


def parse_args():
    p = argparse.ArgumentParser(description="Train NTP model on finite vocab.")
    p.add_argument("--vocab_size", type=int, default=256)
    p.add_argument("--seq_len", type=int, default=64)
    p.add_argument("--n_embd", type=int, default=256)
    p.add_argument("--n_layer", type=int, default=4)
    p.add_argument("--n_head", type=int, default=4)
    p.add_argument("--max_steps", type=int, default=500)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", type=str, default="./output")
    # Wandb arguments
    p.add_argument("--wandb_project", type=str, default="transformers-compression",
                   help="Wandb project name")
    p.add_argument("--wandb_run_name", type=str, default=None,
                   help="Wandb run name (auto-generated if not provided)")
    p.add_argument("--wandb_entity", type=str, default=None,
                   help="Wandb entity (username or team)")
    p.add_argument("--no_wandb", action="store_true",
                   help="Disable wandb logging")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    # Initialize wandb
    use_wandb = not args.no_wandb
    if use_wandb:
        # Only initialize on main process for distributed training
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            run_name = args.wandb_run_name or f"ntp_v{args.vocab_size}_l{args.n_layer}_h{args.n_head}"
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=run_name,
                config={
                    "vocab_size": args.vocab_size,
                    "seq_len": args.seq_len,
                    "n_embd": args.n_embd,
                    "n_layer": args.n_layer,
                    "n_head": args.n_head,
                    "max_steps": args.max_steps,
                    "batch_size": args.batch_size,
                    "learning_rate": args.lr,
                    "seed": args.seed,
                },
            )

    # Synthetic data: (N, seq_len+1) token IDs
    num_samples = max(args.batch_size * 200, 10_000)
    sequences = generate_random_sequences(
        num_sequences=num_samples,
        seq_len=args.seq_len + 1,  # full sequence for causal LM
        vocab_size=args.vocab_size,
        seed=args.seed,
    )
    dataset = FiniteVocabDataset(sequences, vocab_size=args.vocab_size)

    model = create_ntp_model(
        vocab_size=args.vocab_size,
        max_position_embeddings=args.seq_len + 1,
        n_embd=args.n_embd,
        n_layer=args.n_layer,
        n_head=args.n_head,
        use_cache=True,
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        logging_steps=50,
        save_strategy="no",
        report_to="wandb" if use_wandb else "none",
        run_name=args.wandb_run_name,
        seed=args.seed,
        ddp_find_unused_parameters=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    trainer.train()

    print("Training done.")
    # Example: single-step generation with KV cache (on rank 0)
    demo_model = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        _demo_generation(demo_model, args)
        # Close wandb run
        if use_wandb:
            wandb.finish()


def _demo_generation(model, args):
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        prompt = torch.randint(0, args.vocab_size, (1, 4), device=device)
        out = model(prompt, use_cache=True)
        logits = out.logits
        past = out.past_key_values
        next_logits = model(
            prompt[:, -1:],
            past_key_values=past,
            use_cache=True,
        ).logits
        next_id = next_logits[:, -1].argmax(dim=-1)
        print(f"Prompt shape: {prompt.shape}, next token logits shape: {next_logits.shape}, next_id: {next_id.item()}")


if __name__ == "__main__":
    main()
