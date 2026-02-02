import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add parent directory to path to import src modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import generate_random_transition_matrix

def steady_state_dist(P: torch.Tensor) -> torch.Tensor:
    """
    Compute the steady state distribution of a Markov chain.

    Args:
        P: (K, K) transition matrix.

    Returns:
        (K,) steady state distribution.
    """
    K = P.shape[0]
    
    # Compute left-eigenvector of P corresponding to eigenvalue 1
    eigvals, eigvecs = torch.linalg.eig(P.T)
    eigvals, eigvecs = eigvals.real, eigvecs.real
    eigvec = eigvecs[:, eigvals.argmax()]
    return eigvec / eigvec.sum()

def get_tv_distance(p: torch.Tensor, p_ref = None) -> torch.Tensor:
    """
    Compute the total variation distance between two distributions.
    If p_ref is not provided, compute the total variation distance between p and the uniform distribution.
    """
    if p_ref is None:
        p_ref = torch.ones(p.shape[0]) / p.shape[0]
    return torch.sum(torch.abs(p - p_ref)) / 2.0


def main():
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    num_samples = 100
    sizes = [2, 4, 8, 16, 32, 64, 128]
    tv_dists = np.zeros(len(sizes))
    for i, size in enumerate(sizes):
        for _ in range(num_samples):
            P = generate_random_transition_matrix(size)
            pi = steady_state_dist(P)
            tv_dist = get_tv_distance(pi)
            print(f"Size: {size}, Steady state distribution: {pi}, TV distance: {tv_dist}")
            tv_dists[i] += tv_dist
        tv_dists[i] = tv_dists[i] / num_samples / size

    PURPLE = "#741b47"
    GRAY = "#666666"
    font_size = 16
    plt.rcParams.update({'font.size': font_size})
    plt.rcParams.update({'font.family': 'Verdana'})

    # Plot the TV distances on log-log scale
    plt.loglog(sizes, tv_dists, base=2, c=PURPLE, linewidth=2)
    plt.xlabel(r"vocab size ($K$)")
    plt.ylabel(r"normalized tv distance ($\frac{1}{2 K} ||\pi-\text{unif}||_1$)")
    plt.title("concentration of steady state dist.")
    plt.grid(True, linestyle="-", alpha=0.5)
    plt.tight_layout()
    plt.savefig("ss_concentration.png")
    return tv_dists


if __name__ == "__main__":
    main()