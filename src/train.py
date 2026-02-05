"""
Train the NTP model (Hugging Face GPT-2) on finite-vocab sequences with next-token prediction loss.
Uses Hugging Face Trainer for distributed training. For 2 GPUs on one node, run:

    accelerate launch --num_processes 2 train.py [args]

or:

    torchrun --nproc_per_node=2 train.py [args]
"""

import argparse
import logging
import os

import torch
import wandb
from transformers import Trainer, TrainingArguments
from tqdm import tqdm
from dataset import SwitchingMarkovDataset, generate_transition_matrices
from models import create_ntp_model


def parse_args():
    p = argparse.ArgumentParser(description="Train NTP model on finite vocab.")
    p.add_argument("--vocab_size", type=int, default=26)
    p.add_argument("--seq_len", type=int, default=256)
    p.add_argument("--n_embd", type=int, default=128)
    p.add_argument("--n_layer", type=int, default=4)
    p.add_argument("--n_head", type=int, default=4)
    p.add_argument("--max_steps", type=int, default=1000)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--class_type", type=str, default="diff", choices=["same", "diff"])
    p.add_argument("--tilt", type=float, default=0.1)
    p.add_argument("--window_len", type=int, default=256)
    p.add_argument("--num_chains", type=int, default=1000)
    p.add_argument("--num_samples_per_chain", type=int, default=1000)
    p.add_argument("--num_chains_eval", type=int, default=50)
    p.add_argument("--num_samples_per_chain_eval", type=int, default=100)
    p.add_argument("--output_dir", type=str, default="./checkpoints")
    # Wandb arguments
    p.add_argument("--wandb_project", type=str, default="transformers-compression",
                   help="Wandb project name")
    p.add_argument("--wandb_run_name", type=str, default=None,
                   help="Wandb run name (auto-generated if not provided)")
    p.add_argument("--wandb_entity", type=str, default="ml-wave",
                   help="Wandb entity (username or team)")
    p.add_argument("--no_wandb", action="store_true",
                   help="Disable wandb logging")
    p.add_argument("--wandb_id", type=str, default=None,
                   help="Wandb run ID (auto-generated if not provided)")
    return p.parse_args()


def main():
    args = parse_args()

    # Only log from main process in distributed training to avoid duplicate output
    if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
        logging.getLogger().setLevel(logging.WARNING)

    if args.wandb_id is not None:
        resume_from_checkpoint = True
        resume_wandb = "must"
        print(f"Resuming wandb run with ID: {args.wandb_id}")
    else:
        resume_from_checkpoint = False
        resume_wandb = "allow"

    # Initialize wandb
    use_wandb = not args.no_wandb
    if use_wandb:
        # Only initialize on main process for distributed training
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            run_name = args.wandb_run_name or f"ntp_v{args.vocab_size}_l{args.n_layer}_h{args.n_head}_seq{args.seq_len}_window{args.window_len}_class{args.class_type}_tilt{args.tilt}"
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=run_name,
                id=args.wandb_id,
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
                resume=resume_wandb,
            )

    # Create transition matrices and dataset objects
    P_train = generate_transition_matrices(args.num_chains, args.vocab_size, args.tilt)
    P_eval = generate_transition_matrices(args.num_chains_eval, args.vocab_size, args.tilt)

    assert not torch.allclose(P_train[0], P_eval[0]), "Train and eval transition matrices should be different"
    train_dataset = SwitchingMarkovDataset(P_train, args.num_samples_per_chain, args.seq_len, args.window_len, args.seed)
    eval_dataset = SwitchingMarkovDataset(P_eval, args.num_samples_per_chain_eval, args.seq_len, args.window_len, args.seed)

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
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=50,
        report_to="wandb" if use_wandb else "none",
        run_name=args.wandb_run_name,
        seed=args.seed,
        ddp_find_unused_parameters=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

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
