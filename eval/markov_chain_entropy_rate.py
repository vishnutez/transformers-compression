import torch
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import generate_transition_matrices


def markov_chain_entropy_rate(P: torch.Tensor) -> float:
    """
    Compute the entropy rate of a stationary Markov chain.

    The entropy rate is defined as:
        H = - sum_i pi_i * sum_j P(i,j) * log2(P(i,j))

    where pi is the stationary distribution satisfying pi^T P = pi^T.

    Args:
        P: (n, n) row-stochastic transition matrix.

    Returns:
        Entropy rate in bits per symbol.
    """
    # Find stationary distribution via the left eigenvector of P with eigenvalue 1.
    # Equivalently, the right eigenvector of P^T with eigenvalue 1.
    P = P.double()
    eigenvalues, eigenvectors = torch.linalg.eig(P.T)

    # Find the eigenvector corresponding to eigenvalue ≈ 1
    idx = torch.argmin(torch.abs(eigenvalues - 1.0))
    pi = eigenvectors[:, idx].real
    pi = pi / pi.sum()  # normalise to a probability distribution
    pi = pi.abs()       # remove tiny negative artifacts from numerical precision

    # Compute per-state conditional entropy: H_i = - sum_j P(i,j) log2 P(i,j)
    # Use a safe log that maps log(0) -> 0 so that 0*log(0) = 0
    log_P = torch.where(P > 0, torch.log2(P), torch.zeros_like(P))
    entropy_per_state = -torch.sum(P * log_P, dim=1)

    # Entropy rate is the stationary-distribution-weighted average
    entropy_rate = torch.dot(pi, entropy_per_state).item()
    return entropy_rate


# Generate a random transition matrix with a tilt
tilt = 0.1
vocab_size = 26
num_chains = 10
seed = 108
torch.manual_seed(seed)
P = generate_transition_matrices(num_chains, vocab_size, tilt)

avg_ent_rate_pre = 0
avg_ent_rate_post = 0

for i in range(num_chains):
    # Compute the entropy rate of the Markov chain
    ent_rate_pre = markov_chain_entropy_rate(P[i, 0])
    ent_rate_post = markov_chain_entropy_rate(P[i, 1])
    avg_ent_rate_pre += ent_rate_pre
    avg_ent_rate_post += ent_rate_post

avg_ent_rate_pre /= num_chains
avg_ent_rate_post /= num_chains

print(f"Entropy rate (pre): {avg_ent_rate_pre:.4f} bits/symbol")
print(f"Entropy rate (post): {avg_ent_rate_post:.4f} bits/symbol")

# Save the results to a file that can be loaded later maybe numpy array
np.save(f"metrics/avg_ent_rate_pre_t={tilt}_v={vocab_size}_nc={num_chains}.npy", avg_ent_rate_pre)
np.save(f"metrics/avg_ent_rate_post_t={tilt}_v={vocab_size}_nc={num_chains}.npy", avg_ent_rate_post)