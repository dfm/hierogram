# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["test_noiseless"]

import numpy as np
from .hierogram import Hierogram


def test_noiseless():
    np.random.seed(1234)
    x = 5 + np.random.randn(100, 1, 2)
    model = Hierogram(x, [np.linspace(0, 10, 10), np.linspace(0, 10, 10)])
    h = model.optimize(verbose=True)
    y, tmp = np.histogramdd(x[:, 0, :], model.bins)
    assert np.mean((np.exp(h + model.lnvolumes)
                    - y.flatten()) ** 2) < 1e-5
