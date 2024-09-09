from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.io import loadmat
from pyriemann.estimation import Shrinkage


def check_empty(data, field):
    if data[field][0][0].size == 0:
        return np.isnan
    else:
        return data[field][0][0][0][0]


def read_cross_spectra(fname, fmax=49):
    dat = loadmat(fname)['data_struct']
    site = fname.parent.name
    return SimpleNamespace(
        chs=[cc[0] for cc in dat['dnames'][0, 0][0]],
        csd=dat['CrossM'][0, 0][..., :fmax],
        foi=dat['freqrange'][0, 0][0][:fmax],
        spec_foi=dat['Spec_freqrange'][0, 0][0][:fmax],
        age=check_empty(dat, 'age'),
        sex=check_empty(dat, 'sex'),
        n_times=check_empty(dat, 'nt'),
        reference=dat['ref'][0][0][0],
        site=site
    )


def load_dataset(in_path):
    sub_dirs = [pp for pp in in_path.glob('*') if pp.is_dir()]
    dataset = list()
    for sub_dir in sub_dirs:
        for mat_path in sub_dir.glob('*mat'):
            data = read_cross_spectra(mat_path)
            dataset.append(data)
    return dataset


def get_ave_operator(nc=19):
    Id = np.identity(nc)
    A = np.ones((nc, nc)) / nc
    return Id - A


def apply_car(covs):
    nc = covs.shape[-1]
    P = get_ave_operator(nc)
    covs_ref = np.einsum('ij,sfjk,kl->sfil', P, covs, P.T)
    return covs_ref


def apply_shrinkage(covs, shrinkage=1e-5):
    reg = Shrinkage(shrinkage=shrinkage)
    covs_reg = np.zeros_like(covs)
    for freq in range(covs.shape[1]):
        covs_reg[:, freq] = reg.fit_transform(covs[:, freq])
    return covs_reg


def average_columns(X, inds):
    return np.array(X.iloc[:, inds].values.tolist()).mean(axis=1)


def average_wavelets(X, bands):
    band_name, inds = bands
    return pd.DataFrame({band_name: list(average_columns(X, inds))})
