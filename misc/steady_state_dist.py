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
    tv_dists_same_class = np.zeros(len(sizes))
    tv_dists_diff_class = np.zeros(len(sizes))
    for i, size in enumerate(sizes):
        alpha0 = torch.ones(size)
        alpha0[size // 2:] = 0.5
        alpha1 = torch.ones(size)
        alpha1[:size // 2] = 0.5
        for _ in range(num_samples):

            # same class
            P0 = generate_random_transition_matrix(size)
            P1 = generate_random_transition_matrix(size)
            pi0 = steady_state_dist(P0)
            pi1 = steady_state_dist(P1)
            tv_dist = get_tv_distance(pi0, pi1)
            tv_dists_same_class[i] += tv_dist

            # diff class
            P0 = generate_random_transition_matrix(size, alpha0)
            P1 = generate_random_transition_matrix(size, alpha1)
            pi0 = steady_state_dist(P0)
            pi1 = steady_state_dist(P1)
            tv_dist = get_tv_distance(pi0, pi1)
            tv_dists_diff_class[i] += tv_dist

        tv_dists_same_class[i] = tv_dists_same_class[i] / num_samples
        tv_dists_diff_class[i] = tv_dists_diff_class[i] / num_samples
        print(f"Size: {size}, TV distance (same class): {tv_dists_same_class[i]}, TV distance (diff class): {tv_dists_diff_class[i]}")

    PURPLE = "#741b47"
    GRAY = "#666666"
    font_size = 16
    plt.rcParams.update({'font.size': font_size})
    plt.rcParams.update({'font.family': 'Verdana'})

    # Plot the TV distances on log-log scale
    plt.loglog(sizes, tv_dists_same_class, base=2, c=GRAY, linewidth=2, linestyle='--', label='same class')
    plt.loglog(sizes, tv_dists_diff_class, base=2, c=PURPLE, linewidth=2, label='diff class')
    plt.legend(loc='lower left')
    plt.xlabel(r"vocab size ($K$)")
    plt.ylabel(r"normalized tv distance ($\frac{1}{2 K} ||\pi_0-\pi_1||_1$)")
    plt.title("concentration of steady state dists.")
    plt.grid(True, linestyle="-", alpha=0.5)
    plt.tight_layout()
    plt.savefig("ss_concentration_multiple_classes.png")
    return tv_dists


if __name__ == "__main__":
    main()