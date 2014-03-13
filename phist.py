#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

__all__ = ["phist"]

import numpy as np


def phist(x, bins=20):
    """
    Make a histogram of noisy data.

    :param x: ``(K, N)``
        A list of ``N`` posterior samples for measurements of ``K`` examples.

    :param bins:
        Either the integer number of bins or an array of bin edges.

    """
    x = np.atleast_2d(x)

    # Interpret the bin specs.
    try:
        len(bins)
    except TypeError:
        bins = np.linspace(x.min(), x.max(), bins)

    # Make an initial guess at the histogram values.
    theta = np.histogram(np.mean(x, axis=1), bins, density=True)[0]
    m = theta > 0
    theta[m] = np.log(theta[m])
    theta[~m] = -np.inf

    return np.exp(theta), bins


if __name__ == "__main__":
    import matplotlib.pyplot as pl
    K, N = 100, 1000
    means = np.random.randn(K)
    samples = means[:, None] + 1e-4 * np.random.randn(K, N)

    values, bins = phist(samples)

    pl.hist(means, bins, normed=True, histtype="step", color="k", lw=2)
    pl.plot(np.array(zip(bins[:-1], bins[1:])).flatten(),
            np.array(zip(values, values)).flatten(), color="r")
    x = np.linspace(-3, 3, 500)
    pl.plot(x, np.exp(-0.5*x**2)/np.sqrt(2*np.pi))
    pl.savefig("test.png")
