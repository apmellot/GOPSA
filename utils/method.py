import numpy as np
import torch
from joblib import Parallel, delayed

from skada.base import DAEstimator
from skada.utils import torch_minimize
from .optimization import (
    parallel_transport,
    mean_covariance_torch,
    ridge_regression,
    _powm,
    _logm,
    _uvec
)


def score(y_true, y_pred):
    return -np.mean((y_true - y_pred) ** 2)


def apply_parallel_transport(X, sample_domain, X_mean, C_ref, alpha_all):
    X_rct = torch.ones_like(X, dtype=torch.float64)
    for i, k in enumerate(np.unique(sample_domain)):
        mask = sample_domain == k
        alpha = alpha_all[i]
        X_rct[mask] = parallel_transport(
            X[mask], X_mean[k], C_ref, alpha
        )
    return X_rct


def apply_log_map_riemann(X, X_mean, alpha_all, sample_domain):
    z = list()
    indices = list()
    for i, k in enumerate(np.unique(sample_domain)):
        mask = sample_domain == k
        alpha = alpha_all[i]
        X_mean_inv_sqrt = _powm(X_mean[k], -alpha/2)
        X_w = X_mean_inv_sqrt @ X[mask] @ X_mean_inv_sqrt
        log_X_w = _logm(X_w)
        z.append(_uvec(log_X_w))
        indices.append(np.arange(len(X))[mask])
    z = torch.cat(z)
    indices = np.concatenate(indices)
    z = z[np.argsort(indices)]
    return z


class GeodesicOptimization(DAEstimator):
    def __init__(
        self,
        y_mean,
        C_ref,
        lambda_=1,
        max_iter=150,
        n_jobs=1,
        verbose=False,
    ):
        assert isinstance(y_mean, dict), 'y_mean should be a dictionary'
        self.y_mean = y_mean
        self.C_ref = C_ref if isinstance(
            C_ref, torch.Tensor) else torch.tensor(C_ref, dtype=torch.float64)
        self.lambda_ = lambda_
        self.max_iter = max_iter
        self.n_jobs = n_jobs
        self.verbose = verbose

    def _get_infos_to_save(self, sample_domain, alpha_solution,
                           X_mean_domain, X):
        (alpha_solution_dict, X_mean_domain_dict,
         X_parallel_transport_dict) = dict(), dict(), dict()

        for i, k in enumerate(np.unique(sample_domain)):
            # alpha
            alpha_solution_dict[k] = alpha_solution[i].item()

            # mean
            X_mean_domain_dict[k] = X_mean_domain[k].detach().numpy()

            # parallel transported data
            X_mean_inv_sqrt = _powm(X_mean_domain[k], -alpha_solution[i]/2)
            X_parallel_transport_dict[k] = (
                X_mean_inv_sqrt @ X[sample_domain == k] @ X_mean_inv_sqrt
            ).detach().numpy()

        return (alpha_solution_dict,  X_mean_domain_dict,
                X_parallel_transport_dict)

    def fit(self, X, y, sample_domain=None):
        X = torch.tensor(X, dtype=torch.float64)
        y = torch.tensor(y, dtype=torch.float64)
        assert X.ndim == 4, 'X should be a 4D tensor: (n_samples, n_freqs, n_channels, n_channels)'
        # Get source data
        X = X[sample_domain >= 0]
        y = y[sample_domain >= 0]
        sample_domain = sample_domain[sample_domain >= 0]
        # Compute mean covariance
        X_mean_domain = {}
        for k in np.unique(sample_domain):
            mask = sample_domain == k
            X_mean_ = Parallel(n_jobs=self.n_jobs)(
                delayed(mean_covariance_torch)(X[mask, f])
                for f in range(X.shape[1])
            )
            X_mean_domain[k] = torch.stack(X_mean_)

        # Find alpha solutions for source data
        def loss_source(alpha_all):
            alpha_all = torch.sigmoid(alpha_all)
            if self.verbose:
                print("Alpha values :", alpha_all.detach().numpy())
            z = apply_log_map_riemann(X, X_mean_domain,
                                      alpha_all, sample_domain)
            y_hat, beta_hat = ridge_regression(z, y, self.lambda_)
            return torch.linalg.norm(y - y_hat)**2

        # Searching for optimal alpha
        alpha_init = np.zeros(len(np.unique(sample_domain)))
        solution, loss_val = torch_minimize(loss_source, alpha_init, tol=1e-6,
                                            max_iter=self.max_iter)
        alpha_solution = torch.sigmoid(torch.tensor(solution))
        if self.verbose:
            print("Final alpha values :", alpha_solution.detach().numpy())

        # Save
        infos = self._get_infos_to_save(sample_domain, alpha_solution,
                                        X_mean_domain, X)
        (self.source_alpha_solution_, self.source_X_mean_,
         self.source_X_parallel_transport_) = infos

        # Fit the model to get beta_hat
        z = apply_log_map_riemann(X, X_mean_domain,
                                  alpha_solution, sample_domain)
        _, self.beta_hat = ridge_regression(z, y, self.lambda_)

        return self

    def predict(self, X, sample_domain=None):
        X = torch.tensor(X, dtype=torch.float64)

        # Compute mean covariance
        X_mean_domain = {}
        for k in np.unique(sample_domain):
            mask = sample_domain == k
            X_mean_ = Parallel(n_jobs=self.n_jobs)(
                delayed(mean_covariance_torch)(X[mask, f])
                for f in range(X.shape[1])
            )
            X_mean_domain[k] = torch.stack(X_mean_)

        def loss_target(alpha_all):
            alpha_all = torch.sigmoid(alpha_all)
            if self.verbose:
                print("Alpha values :", alpha_all.detach().numpy())
            z = apply_log_map_riemann(X, X_mean_domain, alpha_all,
                                      sample_domain)
            diff = torch.zeros(len(np.unique(sample_domain)))
            for i, k in enumerate(np.unique(sample_domain)):
                mask = sample_domain == k
                # y_hat = z[mask] @ self.beta_hat[:-1] + self.beta_hat[-1]
                y_hat = z[mask] @ self.beta_hat
                diff[i] = torch.mean(y_hat) - self.y_mean[np.abs(k)]
            return torch.linalg.norm(diff)**2

        # Searching for optimal alpha
        alpha_init = np.zeros(len(np.unique(sample_domain)))
        solution, loss_val = torch_minimize(loss_target, alpha_init, tol=1e-6,
                                            max_iter=self.max_iter)
        alpha_solution = torch.sigmoid(torch.tensor(solution))
        if self.verbose:
            print("Final alpha values :", alpha_solution.detach().numpy())

        # Save
        infos = self._get_infos_to_save(sample_domain, alpha_solution,
                                        X_mean_domain, X)
        (self.target_alpha_solution_, self.target_X_mean_,
         self.target_X_parallel_transport_) = infos

        # Make final prediction
        z = apply_log_map_riemann(X, X_mean_domain, alpha_solution,
                                  sample_domain)
        # y_pred = z @ self.beta_hat[:-1] + self.beta_hat[-1]
        y_pred = z @ self.beta_hat

        return y_pred.detach().numpy()

    def score(self, X, y, sample_domain=None):
        y_pred = self.predict(X, sample_domain)
        return score(y, y_pred)
