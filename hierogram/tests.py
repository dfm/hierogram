# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["test_noiseless"]

import numpy as np
from itertools import product
from .hierogram import Hierogram


def test_noiseless():
    np.random.seed(1234)
    x = 5 + np.random.randn(100, 1, 2)
    model = Hierogram(x, [np.linspace(0, 10, 10), np.linspace(0, 10, 10)])
    h = model.optimize(verbose=True)
    y, tmp = np.histogramdd(x[:, 0, :], model.bins)
    assert np.mean((np.exp(h + model.lnvolumes)
                    - y.flatten()) ** 2) < 1e-5


def test_volumes():
    np.random.seed(1234)
    ndim = 3
    x = 5 + np.random.randn(100, 1, ndim)
    bins = [np.sort(np.random.rand(10)) for i in range(ndim)]
    model = Hierogram(x, bins)

    # Compute the volumes in the brute force way.
    lnvolumes = np.empty(model.shape)
    for i, j, k in product(range(9), range(9), range(9)):
        lnvolumes[i, j, k] = (np.log(bins[0][i+1]-bins[0][i]) +
                              np.log(bins[1][j+1]-bins[1][j]) +
                              np.log(bins[2][k+1]-bins[2][k]))

    assert np.mean(lnvolumes - model.lnvolumes.reshape(model.shape)) < 1e-10
