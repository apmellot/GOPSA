import numpy as np
import torch


def _powm(C, alpha):
    eigval, eigvect = torch.linalg.eigh(C)
    C_powm_alpha = eigvect @ torch.diag_embed(
        eigval**(alpha), dim1=-2, dim2=-1
    ) @ torch.transpose(eigvect, -2, -1)
    return C_powm_alpha


def _logm(C):
    eigval, eigvect = torch.linalg.eigh(C)
    C_logm = eigvect @ torch.diag_embed(
        torch.log(eigval), dim1=-2, dim2=-1
    ) @ torch.transpose(eigvect, -2, -1)
    return C_logm


def _expm(C):
    eigval, eigvect = torch.linalg.eigh(C)
    C_expm = eigvect @ torch.diag_embed(
        torch.exp(eigval), dim1=-2, dim2=-1
    ) @ torch.transpose(eigvect, -2, -1)
    return C_expm


def _uvec(X):
    Q = torch.sqrt(torch.tensor(2)) * torch.triu(
        torch.ones(X.shape[-1], X.shape[-1]),
        diagonal=1
    ) + torch.eye(X.shape[-1])
    log_C = X * Q
    mask_triu = torch.triu(torch.ones_like(log_C[0])) == 1
    z = log_C[:, mask_triu]
    return z


def distance_riemann(A, B):
    eigval, eigvect = torch.linalg.eigh(A)
    A_inv_sqrt = eigvect @ torch.diag_embed(
        eigval**(-1/2), dim1=-2, dim2=-1
    ) @ torch.transpose(eigvect, -2, -1)
    B_w = A_inv_sqrt @ B @ A_inv_sqrt
    eigval_B, eigvect_B = torch.linalg.eigh(B_w)
    log_B_w = eigvect_B @ torch.diag_embed(
        torch.log(eigval_B), dim1=-2, dim2=-1
    ) @ torch.transpose(eigvect_B, -2, -1)
    return torch.linalg.norm(log_B_w, dim=(-2, -1))


def mean_covariance_torch(covs, iter=50, tol=1e-9):
    C = covs
    C_mean = torch.mean(C, axis=0)
    nu = 1.0
    tau = np.finfo(np.float64).max
    crit = np.finfo(np.float64).max
    for i in range(iter):
        C_mean_inv_sqrt = _powm(C_mean, -1/2)
        C_mean_sqrt = _powm(C_mean, 1/2)
        log_C = _logm(C_mean_inv_sqrt @ C @ C_mean_inv_sqrt)
        log_C_mean = torch.mean(log_C, axis=0)
        C_mean = C_mean_sqrt @ _expm(log_C_mean) @ C_mean_sqrt
        crit = torch.linalg.norm(log_C_mean, ord='fro')
        h = nu * crit
        if h < tau:
            nu = 0.95 * nu
            tau = h
        else:
            nu = 0.5 * nu
        if crit <= tol or nu <= tol:
            break
    return C_mean


def compute_geodesic(B, A, alpha):
    # Compute eigval for A
    eigval, eigvect = torch.linalg.eigh(A)
    A_sqrt = eigvect @ torch.diag_embed(
        eigval**(1/2), dim1=-2, dim2=-1
    ) @ torch.transpose(eigvect, -2, -1)
    A_inv_sqrt = eigvect @ torch.diag_embed(
        eigval**(-1/2), dim1=-2, dim2=-1
    ) @ torch.transpose(eigvect, -2, -1)
    # Whiten B by A
    B_w = A_inv_sqrt @ B @ A_inv_sqrt
    eigval_B_w, eigvect_B_w = torch.linalg.eigh(B_w)
    # alpha power
    B_w_alpha = eigvect_B_w @ torch.diag_embed(
        eigval_B_w**(alpha), dim1=-2, dim2=-1
    ) @ torch.transpose(eigvect_B_w, -2, -1)
    # Recolor with A
    C_geodesic = A_sqrt @ B_w_alpha @ A_sqrt
    return C_geodesic


def re_center(C, C_mean, C_mean_new):
    eigval_mean, eigvect_mean = torch.linalg.eigh(C_mean)
    eigval_mean_new, eigvect_mean_new = torch.linalg.eigh(C_mean_new)
    C_mean_inv_sqrt = eigvect_mean @ torch.diag_embed(
        eigval_mean**(-1/2), dim1=-2, dim2=-1
    ) @ torch.transpose(eigvect_mean, -2, -1)
    C_mean_new_sqrt = eigvect_mean_new @ torch.diag_embed(
        eigval_mean_new**(1/2), dim1=-2, dim2=-1
    ) @ torch.transpose(eigvect_mean_new, -2, -1)
    C_rct = C_mean_new_sqrt @ C_mean_inv_sqrt @ C @ C_mean_inv_sqrt @ C_mean_new_sqrt
    return C_rct


def parallel_transport(C, C_mean, C_ref, alpha):
    # Parallel transport of C from C_mean to C_ref
    C_mean_new = compute_geodesic(C_mean, C_ref, alpha)
    C_rct = re_center(C, C_mean, C_mean_new)
    return C_rct


def log_map_riemann(C, C_ref):
    eigval, eigvect = torch.linalg.eigh(C_ref)
    C_ref_inv_sqrt = eigvect @ torch.diag_embed(
        eigval**(-1/2), dim1=-2, dim2=-1
    ) @ torch.transpose(eigvect, -2, -1)
    C_w = C_ref_inv_sqrt @ C @ C_ref_inv_sqrt
    eigval_C_w, eigvect_C_w = torch.linalg.eigh(C_w)
    log_C_w = eigvect_C_w @ torch.diag_embed(
        torch.log(eigval_C_w), dim1=-2, dim2=-1
    ) @ torch.transpose(eigvect_C_w, -2, -1)
    Q = torch.sqrt(torch.tensor(2)) * torch.triu(
        torch.ones(log_C_w.shape[-1], log_C_w.shape[-1]),
        diagonal=1
    ) + torch.eye(log_C_w.shape[-1])
    log_C = log_C_w * Q
    mask = torch.triu(torch.ones_like(log_C[0])) == 1
    z = log_C[:, mask]
    return z


def ridge_regression(X, y, lam):
    n, p = X.shape

    if p > n:
        A = X @ torch.transpose(X, -2, -1) + lam * torch.eye(
            n, dtype=torch.float64)
        eigval, eigvect = torch.linalg.eigh(A)
        A_inv = eigvect @ torch.diag_embed(
            eigval**(-1), dim1=-2, dim2=-1
        ) @ torch.transpose(eigvect, -2, -1)
        beta_hat = torch.transpose(X, -2, -1) @ A_inv @ y
    else:
        A = torch.transpose(X, -2, -1) @ X + lam * torch.eye(
            p, dtype=torch.float64)
        eigval, eigvect = torch.linalg.eigh(A)
        A_inv = eigvect @ torch.diag_embed(
            eigval**(-1), dim1=-2, dim2=-1
        ) @ torch.transpose(eigvect, -2, -1)
        beta_hat = A_inv @ torch.transpose(X, -2, -1) @ y

    y_hat = X @ beta_hat

    return y_hat, beta_hat
