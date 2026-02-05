from torch.utils.data import Dataset
import torch
from math import floor, ceil


def generate_random_transition_matrix(vocab_size: int, alpha: torch.Tensor | None = None) -> torch.Tensor:
    """Generate a random row-stochastic Markov transition matrix (vocab_size, vocab_size)."""
    # Sample each row from Dirichlet(alpha) so rows sum to 1
    if alpha is None:
        alpha = torch.ones(vocab_size)
    else:
        assert alpha.size(0) == vocab_size, "alpha must be of size (vocab_size,)"
    dist = torch.distributions.Dirichlet(alpha)
    return dist.sample((vocab_size,))


def generate_transition_matrices(num_chains: int, vocab_size: int, tilt: float) -> torch.Tensor:
    """
    Generate transition matrices for a switching Markov chain.
    """
    P = torch.zeros(num_chains, 2, vocab_size, vocab_size)
    for i in range(num_chains):
        alpha0 = torch.ones(vocab_size)
        alpha1 = torch.ones(vocab_size)
        alpha0[vocab_size // 2:] = tilt
        alpha1[:vocab_size // 2] = tilt
        P[i, 0] = generate_random_transition_matrix(vocab_size, alpha=alpha0)
        P[i, 1] = generate_random_transition_matrix(vocab_size, alpha=alpha1)
    return P



class SwitchingMarkovDataset(Dataset):
    def __init__(self, 
                P: torch.Tensor,
                num_samples_per_chain: int, 
                seq_len: int, 
                window_len: int,
                base_seed: int):
        """
        P: torch.Tensor [C, 2, V, V] with rows stochastic (P0,P1), float32
        """
        self.P = P.contiguous()
        self.num_chains = P.shape[0]
        self.vocab_size = P.shape[-1]
        self.num_samples_per_chain = num_samples_per_chain
        self.seq_len = seq_len
        self.mid = seq_len // 2
        self.window_len = window_len
        self.num_samples = self.num_chains * self.num_samples_per_chain
        self.base_seed = int(base_seed)
        

    def __len__(self):
        return self.num_samples

    def _chain_id(self, idx: int) -> int:
        # Option 1: perfectly balanced coverage
        return idx // self.num_samples_per_chain

        # Option 2 (if you want to decorrelate idx ordering but still deterministic):
        # return (idx * 2654435761) % self.C  # Knuth hash

    def __getitem__(self, idx: int):
        c = self._chain_id(idx)

        # Deterministic RNG per sample (safe with multi-worker + DDP)
        g = torch.Generator()
        g.manual_seed(self.base_seed + idx)

        # Generate a switch time uniformly from the window around the middle of the sequence.
        switch_t = torch.randint(floor(self.mid - self.window_len / 2), ceil(self.mid + self.window_len / 2), (1,), generator=g).item()

        # Generate a Markov chain sequence using P0 until switch_t then P1
        vocab_size = self.P.shape[-1]
        x = torch.empty(self.seq_len, dtype=torch.long)

        # Initial token (choose your own init distribution)
        x0 = torch.randint(0, vocab_size, (1,), generator=g).item()
        x[0] = x0

        for t in range(1, self.seq_len):
            P_t = self.P[c, 0] if t <= switch_t else self.P[c, 1]
            probs = P_t[x[t-1]]                      # [vocab_size]
            x[t] = torch.multinomial(probs, num_samples=1, generator=g).item()

        return {
            "input_ids": x,          # [seq_len] long
            "labels": x.clone(),     # if doing LM-style training
            "chain_id": torch.tensor(c, dtype=torch.int64),
            "switch_t": torch.tensor(switch_t, dtype=torch.int64),
        }