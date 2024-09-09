import numpy as np
from scipy.linalg import expm
from pyriemann.utils.base import powm
from sklearn.utils import check_random_state

from skada.datasets import DomainAwareDataset


def _generate_X_y(n_sources, A, powers, sigma_p, beta, sigma_n, sigma_y, rng):
    n_matrices = len(powers)
    n_dim = A.shape[0]

    # Generate covariances
    Cs = np.zeros((n_matrices, n_dim, n_dim))
    for i in range(n_matrices):
        Cs[i, :n_sources, :n_sources] = np.diag(
            powers[i])**sigma_p  # set diag sources
        N_i = sigma_n * rng.randn(n_dim - n_sources, n_dim - n_sources)
        Cs[i, n_sources:, n_sources:] = N_i.dot(N_i.T)  # fill the noise block
    X = A @ Cs @ A.T

    # Generate y
    y = np.log(powers) @ beta
    y += sigma_y * rng.randn(n_matrices)

    return X, y


def simulate_reg_source_target(
    n_source_domains=1,
    n_matrices=1000,
    n_dim=2,
    n_sources=2,
    mixing_difference=0,
    shift_powers=0,
    random_state=42,
):
    """Generate simulated source and target datasets for a
       regression situation.

    Parameters
    ----------
    n_source_domains: int
        Number of source domains.
    n_matrices: int
        Number of matrices in each class.
    n_dim: int
        Dimension of the matrices.
    n_sources: int
        Number of signal sources.
    mixing_difference: float
        Should have a value between 0 and 1. If mixing_difference = 0,
        A_target = A_source. If mixing_difference = 1, A_target is a
        completely different matrix.
    shift_powers: float
        Shift the powers between domains.
    random_state: int
        Random seed used to initialize the pseudo-random number generator.

    Returns
    ----------
    X_source: ndarray shape (2*n_matrices, n_dim, n_dim)
        Matrices of the source dataset.
    y_source: list
        Labels of the source matrices.
    X_target: ndarray shape (2*n_matrices, n_dim, n_dim)
        Matrices of the target dataset.
    y_target: list
        Labels of the target matrices.
    """
    rng = check_random_state(random_state)
    n_domains = n_source_domains + 1

    # Generate A
    A = rng.randn(n_dim, n_dim)

    # Generate powers
    powers = rng.uniform(0.01, 1, size=(n_domains, n_matrices, n_sources))

    # Center the log-powers
    log_powers = np.log(powers)
    log_powers -= log_powers.mean(axis=1, keepdims=True)
    powers = np.exp(log_powers)

    # Unit norm
    powers /= np.linalg.norm(powers)

    # Generate regression coefficients
    beta = rng.randn(n_sources)

    # Generate source and target datasets
    dataset = DomainAwareDataset()
    for i in range(n_domains):
        # Generate A for the domain
        Pv = rng.randn(n_dim, n_dim)  # create random tangent vector
        Pv = (Pv + Pv.T) / 2  # symmetrize
        Pv /= np.linalg.norm(Pv)  # normalize
        P = expm(Pv)  # take it back to the SPD manifold
        M = powm(P, alpha=mixing_difference)  # control distance to identity
        A_domain = M @ A

        # Shift the log-powers
        powers_domain = (powers[i]) ** (1 + i * shift_powers)

        # Generate X and y
        X, y = _generate_X_y(
            n_sources,
            A_domain,
            powers_domain,
            beta=beta,
            sigma_p=1,
            sigma_n=0,
            sigma_y=0,
            rng=rng,
        )

        # Add the dataset
        dataset.add_domain(X, y, domain_name=str(i))

    # Pack the source and target datasets
    # pick a random target domain
    target_domain = rng.choice(n_domains)
    target_domains = [str(target_domain)]
    source_domains = [str(i) for i in range(n_domains) if i != target_domain]
    X, y, sample_domain = dataset.pack(
        as_sources=source_domains,
        as_targets=target_domains,
    )

    return X, y, sample_domain
