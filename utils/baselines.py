import numpy as np
import pandas as pd
from pathlib import Path
from joblib import Parallel, delayed
from lightning import Trainer
import torch
from torch.utils.data import TensorDataset, DataLoader
from pyriemann.tangentspace import TangentSpace
from skada.base import DAEstimator
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedShuffleSplit

from .green_utils import get_train_test_loaders


def score(y_true, y_pred):
    return -np.mean((y_true - y_pred) ** 2)


class Dummy(DAEstimator):
    def __init__(self, y_mean):
        # check if y_mean is a dictionary
        assert isinstance(y_mean, dict), 'y_mean should be a dictionary'
        self.y_mean = y_mean

    def fit(self, X, y=None, sample_domain=None):
        self.fitted_ = True
        return self

    def predict(self, X, sample_domain=None):
        y_pred = np.zeros(X.shape[0])
        for domain in np.unique(sample_domain):
            mask = sample_domain == domain
            y_pred[mask] = self.y_mean[np.abs(domain)]

        return y_pred

    def score(self, X, y, sample_domain=None):
        # Predict
        y_pred = self.predict(X, sample_domain)

        # Return score
        return score(y, y_pred)


class TSRidge(DAEstimator):
    def __init__(self, recenter, rescale, fit_intercept_per_domain,
                 y_mean=None, lambda_=1, n_jobs=1):
        self.recenter = recenter
        self.rescale = rescale
        self.fit_intercept_per_domain = fit_intercept_per_domain
        if fit_intercept_per_domain:
            assert isinstance(y_mean, dict), 'y_mean should be a dictionary'
        self.y_mean = y_mean
        self.lambda_ = lambda_
        self.n_jobs = n_jobs

    def fit(self, X, y, sample_domain=None):
        assert X.ndim == 4, 'X should be a 4D tensor: (n_samples, n_freqs, n_channels, n_channels)'

        # Get source data
        X = X[sample_domain >= 0]
        y = y[sample_domain >= 0]
        sample_domain = sample_domain[sample_domain >= 0]

        # Map to tangent space
        n_samples, n_freqs, _, _ = X.shape
        if self.recenter:
            Z = list()
            indices = list()
            for k in np.unique(sample_domain):
                mask = sample_domain == k
                Z_freq = Parallel(n_jobs=self.n_jobs)(
                    delayed(
                        TangentSpace(metric='riemann',
                                     tsupdate=False).fit_transform
                    )(X[mask, freq]) for freq in range(n_freqs)
                )
                if self.rescale:
                    for freq in range(n_freqs):
                        Z_freq[freq] /= np.sqrt(
                            np.linalg.norm(
                                Z_freq[freq])**2 / Z_freq[freq].shape[0]
                        )
                Z_freq = np.stack(Z_freq).transpose(1, 0, 2)
                Z.append(Z_freq)
                indices.append(np.arange(n_samples)[mask])
            Z = np.concatenate(Z)
            indices = np.concatenate(indices)
            Z = Z[np.argsort(indices)]
        else:
            self.ts_ = Parallel(n_jobs=self.n_jobs)(
                delayed(
                    TangentSpace(metric='riemann', tsupdate=False).fit
                )(X[:, freq]) for freq in range(n_freqs)
            )
            Z = Parallel(n_jobs=self.n_jobs)(
                delayed(
                    ts.transform
                )(X[:, freq]) for freq, ts in enumerate(self.ts_)
            )
            Z = np.stack(Z).transpose(1, 0, 2)
        Z = Z.reshape(n_samples, -1)

        # Fit intercept
        if not self.fit_intercept_per_domain:
            self.intercept_ = np.mean(y)

        # Train Ridge
        if self.fit_intercept_per_domain:
            y_centered = np.zeros(y.shape)
            for k in np.unique(sample_domain):
                mask = sample_domain == k
                y_centered[mask] = y[mask] - self.y_mean[np.abs(k)]
        else:
            y_centered = y - self.intercept_
        self.ridge_ = Ridge(alpha=self.lambda_, fit_intercept=False)
        self.ridge_.fit(Z, y_centered)

        return self

    def predict(self, X, sample_domain=None):
        # Map to tangent space
        n_samples, n_freqs, _, _ = X.shape
        if self.recenter:
            Z = list()
            indices = list()
            for k in np.unique(sample_domain):
                mask = sample_domain == k
                Z_freq = Parallel(n_jobs=self.n_jobs)(
                    delayed(
                        TangentSpace(metric='riemann',
                                     tsupdate=False).fit_transform
                    )(X[mask, freq]) for freq in range(n_freqs)
                )
                if self.rescale:
                    for freq in range(n_freqs):
                        Z_freq[freq] /= np.sqrt(
                            np.linalg.norm(
                                Z_freq[freq])**2 / Z_freq[freq].shape[0]
                        )
                Z_freq = np.stack(Z_freq).transpose(1, 0, 2)
                Z.append(Z_freq)
                indices.append(np.arange(n_samples)[mask])
            Z = np.concatenate(Z)
            indices = np.concatenate(indices)
            Z = Z[np.argsort(indices)]
        else:
            Z = Parallel(n_jobs=self.n_jobs)(
                delayed(
                    ts.transform
                )(X[:, freq]) for freq, ts in enumerate(self.ts_)
            )
            Z = np.stack(Z).transpose(1, 0, 2)
        Z = Z.reshape(n_samples, -1)

        # Predict
        y_pred = self.ridge_.predict(Z)
        if self.fit_intercept_per_domain:
            for k in np.unique(sample_domain):
                mask = sample_domain == k
                y_pred[mask] += self.y_mean[np.abs(k)] - np.mean(y_pred[mask])
        else:
            y_pred += self.intercept_

        return y_pred

    def score(self, X, y, sample_domain=None):
        # Predict
        y_pred = self.predict(X, sample_domain)

        # Return score
        return score(y, y_pred)


class GREEN(DAEstimator):
    def __init__(self, model, random_state):
        self.model = model
        self.random_state = random_state
        torch.manual_seed(self.random_state)
        ckpt_path: str = "checkpoints"
        if Path(
            ckpt_path + "/preds.csv").exists() or Path(
                ckpt_path + "/y_pred_proba.csv").exists():
            print(f"Seed {self.random_state} already trained")
            return None

        self.trainer = Trainer(max_epochs=30,
                               log_every_n_steps=1,
                               default_root_dir=ckpt_path,
                               precision=64,
                               enable_progress_bar=False
                               )

    def fit(self, X, y, sample_domain=None):
        X = torch.tensor(X, dtype=torch.float64)
        y = torch.tensor(y, dtype=torch.float64)
        dataset = TensorDataset(X, y)

        cv = StratifiedShuffleSplit(n_splits=1, test_size=0.2,
                                    random_state=self.random_state)
        train_indices, test_indices = next(cv.split(X, sample_domain))

        (train_dataloader,
         test_dataloader) = get_train_test_loaders(dataset,
                                                   train_indices,
                                                   test_indices,
                                                   batch_size=128,
                                                   num_workers=1,
                                                   final_val=False,
                                                   shuffle=False)
        del X, y, dataset
        self.trainer.fit(
            model=self.model,
            train_dataloaders=train_dataloader,
            val_dataloaders=test_dataloader
        )

    def predict(self, X, sample_domain=None):
        X = torch.tensor(X, dtype=torch.float64)
        y = torch.ones(X.shape[0])
        dataset = TensorDataset(X, y)
        final_loader = DataLoader(dataset,
                                  batch_size=1,
                                  num_workers=1,
                                  shuffle=False,
                                  multiprocessing_context='fork')
        del X, dataset
        y_pred = pd.concat(
            self.trainer.predict(self.model,
                                 final_loader)).y_pred.values
        return y_pred

    def score(self, X, y, sample_domain=None):
        # Predict
        y_pred = self.predict(X, sample_domain)
        # Return score
        return score(y, y_pred)
