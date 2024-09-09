import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import itertools
from joblib import Parallel, delayed

from skada.datasets import DomainAwareDataset
from sklearn.model_selection import (StratifiedShuffleSplit,
                                     GridSearchCV, KFold)
import coffeine
from utils import preprocessing
from utils.baselines import (
    Dummy,
    TSRidge,
    GREEN
)
from utils.green_utils import get_green_g2, GreenRegressorLM
from utils.method import GeodesicOptimization

import sklearn
sklearn.set_config(enable_metadata_routing=True)

parser = argparse.ArgumentParser(description="Run geodesic optimization")
parser.add_argument('-s', '--seed', default=42,
                    help='Random seed for CV')
args = parser.parse_args()
random_state = int(args.seed)
print('Random state :', random_state)


def get_data():
    # Read dataset and make covs
    in_path = Path('path_to_dataset')
    dataset = preprocessing.load_dataset(in_path)

    covs_ = np.transpose([dd.csd.real for dd in dataset], [0, 3, 1, 2])

    age_ = np.array([np.float64(dd.age) for dd in dataset])
    site_ = np.array([dd.site for dd in dataset])

    # Remove subjects without age info
    covs = covs_[~ np.isnan(age_)]
    sample_sites = site_[~ np.isnan(age_)]
    age = age_[~ np.isnan(age_)]
    y = age

    # Apply common average reference
    covs_ref = preprocessing.apply_car(covs)
    # Regularization to have SPD matrices
    covs_ref_reg = preprocessing.apply_shrinkage(covs_ref)
    # Build coffeine dataframe
    C_df = coffeine.make_coffeine_data_frame(covs_ref_reg)
    # Compute log powers
    log_power = coffeine.make_filter_bank_transformer(
        names=C_df.columns[:49],
        method='log_diag',
    ).fit_transform(C_df.iloc[:, :49])
    log_power = log_power.reshape(len(log_power), -1, 19)
    # Compute the global scale factor...
    log_gsf = np.sum(
        log_power, axis=1).sum(axis=1) / np.multiply(*log_power.shape[1:])
    # ... and apply it
    covs_gsf = covs_ref_reg / np.exp(log_gsf)[
        :, np.newaxis, np.newaxis, np.newaxis
    ]

    return covs_gsf, y, sample_sites


def save_results(y_pred, site_source_names_key, method, site_target,
                 y_target, s):
    df_results = []
    for i in range(len(y_pred)):
        df_results.append({
            'sites_source_index': site_source_names_key,
            'method': method,
            'site_target': site_target[i],
            'y_pred': y_pred[i].item(),
            'y_true': y_target[i].item(),
            'split_index': s
            }
        )
    df_results = pd.DataFrame(df_results)
    return df_results


def run_all(
    dataset,
    y_mean,
    site_source_names_key,
    site_source_names,
    site_names,
    method,
    random_state,
    n_jobs
):
    print(f"Method: {method}")
    site_target_names = list(set(site_names) - set(site_source_names))
    X, y, sample_domain = dataset.pack(as_sources=site_source_names,
                                       as_targets=site_target_names)
    X_source = X[sample_domain > 0]
    y_source = y[sample_domain > 0]
    sample_domain_source = sample_domain[sample_domain > 0]
    X_target = X[sample_domain < 0]
    y_target = y[sample_domain < 0]
    sample_domain_target = sample_domain[sample_domain < 0]
    del X, y, sample_domain

    # Associate y_mean to the domain indices
    y_mean = {dataset.domain_names_[k]: v for k, v in y_mean.items()}

    # Get one split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5,
                                 random_state=random_state)
    train_index, test_index = next(sss.split(X_target, sample_domain_target))

    # Compute y_mean target on training set
    for k in np.unique(sample_domain_target):
        sample_domain_target_train = sample_domain_target[train_index]
        y_target_train = y_target[train_index]
        mask = sample_domain_target_train == k
        y_mean[np.abs(k)] = np.mean(y_target_train[mask])

    X_target, y_target = X_target[test_index], y_target[test_index]
    sample_domain_target = sample_domain_target[test_index]

    # Get sample test names for later saving
    site_indices_to_names = {v: k for k, v in dataset.domain_names_.items()}
    sample_domain_target_names = [
        site_indices_to_names[np.abs(i)] for i in sample_domain_target
    ]
    del dataset

    # Train and test depending on the method
    results = list()
    lambda_ = 100
    if method == 'dummy':
        regressor = Dummy(y_mean=y_mean)
    elif method == 'baseline_no_recenter':
        regressor = TSRidge(recenter=False, rescale=False,
                            fit_intercept_per_domain=False,
                            lambda_=lambda_, n_jobs=n_jobs)
    elif method == 'baseline_green':
        green_g2 = get_green_g2(
            n_ch=X_source.shape[-1],
            n_freqs=X_source.shape[1],
            orth_weights=True,
            dropout=.5,
            hidden_dim=[64, 32],
            logref='logeuclid',
            bi_out=[X_source.shape[-1]-1],
            dtype=torch.float64,
            out_dim=1
        )
        model = GreenRegressorLM(model=green_g2, lr=1e-1,
                                 data_type=torch.float64)
        regressor = GREEN(model, random_state=random_state)
    elif method == 'baseline_recenter':
        regressor = TSRidge(recenter=True, rescale=False,
                            fit_intercept_per_domain=False,
                            lambda_=lambda_, n_jobs=n_jobs)
    elif method == 'baseline_rescale':
        regressor = TSRidge(recenter=True, rescale=True,
                            fit_intercept_per_domain=False,
                            lambda_=lambda_, n_jobs=n_jobs)
    elif method == 'baseline_fit_intercept':
        regressor = TSRidge(
            recenter=False, rescale=False, fit_intercept_per_domain=True,
            y_mean=y_mean, lambda_=lambda_, n_jobs=n_jobs
        )
    elif method == 'geodesic_optim':
        C_ref = np.eye(X_source.shape[-1])
        regressor = GeodesicOptimization(
            y_mean=y_mean, C_ref=torch.tensor(C_ref, dtype=torch.float64),
            lambda_=lambda_)

    # Grid search for the best lambda
    if method != 'dummy' and method != 'baseline_green':
        lambda_grid = np.logspace(0, 5, 6)
        regressor = GridSearchCV(
            regressor,
            {'lambda_': lambda_grid},
            refit=True,
            cv=KFold(n_splits=5, shuffle=True, random_state=random_state),
            n_jobs=30
        )

    # Fit
    regressor.fit(X_source, y_source, sample_domain=sample_domain_source)
    if method != 'dummy' and method != 'baseline_green':
        print(f"{method}: best lambda: {regressor.best_params_['lambda_']}")
        regressor = regressor.best_estimator_

    # Predict
    y_pred = regressor.predict(X_target, sample_domain=sample_domain_target)

    # Save results
    results = save_results(
        y_pred,
        site_source_names_key,
        method,
        sample_domain_target_names,
        y_target,
        s=random_state
    )
    results = pd.concat([results])

    return results


DEBUG = False
N_JOBS = 16

covs_gsf, y, sample_sites = get_data()

# Compute y_mean
y_mean = dict()
for site in np.unique(sample_sites):
    y_mean[site] = np.mean(y[sample_sites == site])

# Create a skada dataset
dataset = DomainAwareDataset()
for domain in np.unique(sample_sites):
    mask = sample_sites == domain
    dataset.add_domain(covs_gsf[mask], y[mask], domain)

site_names = np.unique(sample_sites)
site_source_names = {
    1: ['Barbados', 'Chongqing', 'Germany', 'Switzerland'],
    2: ['Bern', 'CHBMP', 'Switzerland'],
    3: ['Barbados', 'Colombia', 'Germany'],
    4: ['Malaysia', 'Russia', 'Cuba2003', 'Switzerland'],
    5: ['Barbados', 'Bern', 'Chongqing', 'Colombia', 'Cuba90', 'Germany',
        'Russia']
}

methods = ['dummy', 'baseline_no_recenter', 'baseline_green',
           'baseline_recenter', 'baseline_rescale',
           'baseline_fit_intercept', 'geodesic_optim']

if DEBUG:
    N_JOBS = 1
    methods = ['baseline_green']
    site_source_names = {
        1: ['Barbados', 'Chongqing', 'Germany', 'Switzerland']
    }

all_results = Parallel(n_jobs=N_JOBS)(
    delayed(run_all)(
        dataset,
        y_mean,
        site_source_names_key,
        site_source_names[site_source_names_key],
        site_names,
        method,
        random_state,
        n_jobs=N_JOBS
    ) for site_source_names_key, method in itertools.product(
        site_source_names, methods
    )
)

all_results = pd.concat(all_results)

if DEBUG:
    all_results.to_csv('./results/geodesic_optimization_results_debug.csv')
else:
    all_results.to_csv(
        f'./results/geodesic_optimization_results_{random_state}.csv'
    )
