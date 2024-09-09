from pathlib import Path
import itertools
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import torch

from utils.baselines import (
    Dummy,
    TSRidge,
    GREEN
)
from utils.green_utils import get_green_g2, GreenRegressorLM
from utils.method import GeodesicOptimization
from utils.simulation import simulate_reg_source_target


RESULTS_FOLDER = Path('./results/')
RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)

N_JOBS = -1
N_REPEATS = 100
N_SOURCE_DOMAINS = 5
N_DIM = 5
N_SOURCES = N_DIM
N_MATRICES = 300

DEBUG = False

SCENARIOS = ["shift_X", "shift_Y", "shift_XY"]
METHODS = ['dummy', 'baseline_no_recenter', 'baseline_green',
           'baseline_recenter', 'baseline_rescale',
           'baseline_fit_intercept', 'geodesic_optim']

if DEBUG:
    N_JOBS = -1
    N_REPEATS = 10
    SCENARIOS = ["shift_X", "shift_Y", "shift_XY"]
    METHODS = ['dummy', 'baseline_rescale', 'baseline_green']

rng = np.random.RandomState(42)
RANDOM_STATES = rng.randint(0, 10000, N_REPEATS)


def run_one(
    scenario,
    methods,
    n_source_domains,
    n_matrices,
    n_dim,
    n_sources,
    mixing_difference,
    shift_powers,
    random_state,
):
    # Generate data
    X, y, sample_domain = simulate_reg_source_target(
        n_source_domains=n_source_domains,
        n_matrices=n_matrices,
        n_dim=n_dim,
        n_sources=n_sources,
        mixing_difference=mixing_difference,
        shift_powers=shift_powers,
        random_state=random_state,
    )

    # Reshape X to match
    # (n_samples, n_freqs, n_channels, n_channels)
    X = X[:, np.newaxis, :, :]

    # Compute y_mean (1=source, 2=target)
    y_mean = dict()
    for domain in np.unique(sample_domain):
        y_mean[np.abs(domain)] = np.mean(y[sample_domain == domain])
    print(f"y_mean: {y_mean}")

    if scenario == "shift_X":
        parameter = mixing_difference
    elif scenario == "shift_Y":
        parameter = shift_powers
    elif scenario == "shift_XY":
        parameter = np.mean([mixing_difference, shift_powers])

    # Split source and target
    X_source = X[sample_domain > 0]
    y_source = y[sample_domain > 0]
    sample_domain_source = sample_domain[sample_domain > 0]
    X_target = X[sample_domain < 0]
    y_target = y[sample_domain < 0]
    sample_domain_target = sample_domain[sample_domain < 0]

    results = None
    for method in methods:
        print('Scenario: ', scenario,
              ', Method: ', method,
              ', Parameter value: ', parameter)

        lambda_ = 1e-3
        if method == 'dummy':
            regressor = Dummy(y_mean=y_mean)
        elif method == 'baseline_no_recenter':
            regressor = TSRidge(
                recenter=False, rescale=False,
                fit_intercept_per_domain=False,
                lambda_=lambda_, n_jobs=1
            )
        elif method == 'baseline_green':
            green_g2 = get_green_g2(
                n_ch=n_dim,
                n_freqs=1,
                orth_weights=True,
                dropout=0.5,
                hidden_dim=[64, 32],
                logref='logeuclid',
                bi_out=[n_dim - 1],
                dtype=torch.float64,
                out_dim=1
            )
            model = GreenRegressorLM(model=green_g2, lr=1e-1,
                                     data_type=torch.float64)
            regressor = GREEN(model, random_state=random_state)
        elif method == 'baseline_recenter':
            regressor = TSRidge(
                recenter=True, rescale=False,
                fit_intercept_per_domain=False,
                lambda_=lambda_, n_jobs=1
            )
        elif method == 'baseline_rescale':
            regressor = TSRidge(
                recenter=True, rescale=True,
                fit_intercept_per_domain=False,
                lambda_=lambda_, n_jobs=1
            )
        elif method == 'baseline_fit_intercept':
            regressor = TSRidge(
                recenter=False, rescale=False,
                fit_intercept_per_domain=True,
                y_mean=y_mean, lambda_=lambda_, n_jobs=1
            )
        elif method == 'geodesic_optim':
            regressor = GeodesicOptimization(
                y_mean=y_mean, C_ref=np.eye(n_dim), lambda_=lambda_
            )

        # Fit
        regressor.fit(X_source, y_source, sample_domain=sample_domain_source)

        # Predict
        y_pred = regressor.predict(
            X_target, sample_domain=sample_domain_target)

        # Save results
        new_results = save_results(
            y_pred=y_pred,
            y_target=y_target,
            method=method,
            scenario=scenario,
            parameter=parameter,
            random_state=random_state,
        )
        results = new_results if results is None else pd.concat([results,
                                                                 new_results])

    results = pd.concat([results])

    return results


def save_results(y_pred, y_target, method, scenario, parameter, random_state):
    df_results = list()
    for i in range(len(y_pred)):
        df_results.append({
            'y_pred': y_pred[i].item(),
            'y_true': y_target[i].item(),
            'method': method,
            'scenario': scenario,
            'parameter': parameter,
            'random_state': random_state
            }
        )
    df_results = pd.DataFrame(df_results)
    return df_results


def run_scenarios_methods(
    scenario,
    methods,
    n_source_domains,
    n_matrices,
    n_dim,
    n_sources,
    random_state
):
    mixing_difference_list = np.linspace(0, 0.5, 5)
    shift_powers_list = np.linspace(0, 0.17, 5)

    if scenario == "shift_X":
        shift_powers = 0

        results = [
            run_one(
                scenario=scenario,
                methods=methods,
                n_source_domains=n_source_domains,
                n_matrices=n_matrices,
                n_dim=n_dim,
                n_sources=n_sources,
                mixing_difference=mixing_difference,
                shift_powers=shift_powers,
                random_state=random_state,
            ) for mixing_difference in mixing_difference_list
        ]

    elif scenario == "shift_Y":
        mixing_difference = 0

        results = [
            run_one(
                scenario=scenario,
                methods=methods,
                n_source_domains=n_source_domains,
                n_matrices=n_matrices,
                n_dim=n_dim,
                n_sources=n_sources,
                mixing_difference=mixing_difference,
                shift_powers=shift_powers,
                random_state=random_state,
            ) for shift_powers in shift_powers_list
        ]

    elif scenario == "shift_XY":
        results = [
            run_one(
                scenario=scenario,
                methods=methods,
                n_source_domains=n_source_domains,
                n_matrices=n_matrices,
                n_dim=n_dim,
                n_sources=n_sources,
                mixing_difference=mixing_difference,
                shift_powers=shift_powers,
                random_state=random_state,
            ) for (mixing_difference, shift_powers) in zip(
                mixing_difference_list, shift_powers_list)
        ]

    else:
        raise ValueError("Unknown scenario")

    return pd.concat(results)


results = Parallel(n_jobs=N_JOBS)(
    delayed(run_scenarios_methods)(
        scenario=scenario,
        methods=METHODS,
        n_source_domains=N_SOURCE_DOMAINS,
        n_matrices=N_MATRICES,
        n_dim=N_DIM,
        n_sources=N_SOURCES,
        random_state=random_state
    )
    for scenario, random_state in itertools.product(
        SCENARIOS, RANDOM_STATES
    )
)
results = pd.concat(results)

if DEBUG:
    results.to_csv(RESULTS_FOLDER / 'simulated_expe_debug.csv')
else:
    results.to_csv(RESULTS_FOLDER / 'simulated_expe.csv')
