#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["phist"]

import numpy as np
import scipy.optimize as op
from scipy.misc import logsumexp


def phist(x, bins=10, lnpriors=None, verbose=False):
    """
    Make a histogram of noisy data.

    :param x: ``(K, N)``
        A list of ``N`` posterior samples for measurements of ``K`` examples.

    :param bins:
        Either the integer number of bins or an array of bin edges.

    :param lnpriors: ``(K, N)`` or ``None``
        If not ``None``, this should be an array with the log prior evaluated
        at the samples given in ``x``.

    """
    x = np.atleast_2d(x)

    # Interpret the bin specs.
    try:
        len(bins)
    except TypeError:
        bins = np.linspace(x.min(), x.max(), bins)

    # Pre-compute the bin indices of the samples.
    inds = np.digitize(x.flatten(), bins).reshape(x.shape)

    # Generate a prior array if one wasn't provided.
    if lnpriors is None:
        lnpriors = np.zeros_like(x)

    # Make an initial guess at the histogram values.
    theta = np.array(np.histogram(np.mean(x, axis=1), bins)[0], dtype=float)
    theta /= np.sum(theta)
    m = theta > 0
    theta[m] = np.log(theta[m])
    theta[~m] = -np.inf

    # # FIXME: Start at a dumb place.
    # theta = np.zeros_like(theta) - np.log(len(theta))

    # Compute the maximum likelihood histogram.
    results = op.minimize(_objective, theta, args=(inds, lnpriors),
                          method="L-BFGS-B",
                          bounds=[(None, 0)]*len(theta))
    if verbose:
        print(results)

    # Normalize the distribution to a density.
    density = np.exp(results.x)
    density /= np.sum(density*(bins[1:] - bins[:-1]))

    return density, bins


def _objective(p, inds, lnpriors):
    theta = np.empty(len(p)+2)
    theta[1:-1] = p - logsumexp(p)
    theta[0] = -np.inf
    theta[-1] = -np.inf
    return -np.sum(logsumexp(theta[inds] - lnpriors, axis=1))


if __name__ == "__main__":
    import matplotlib.pyplot as pl
    K, N = 500, 1000
    means = np.random.randn(K)
    samples = means[:, None] + 1e-1 * np.random.randn(K, N)

    values, bins = phist(samples, verbose=True)

    pl.hist(means, bins, normed=True, histtype="step", color="k", lw=2)
    pl.plot(np.array(zip(bins[:-1], bins[1:])).flatten(),
            np.array(zip(values, values)).flatten(), color="r")
    x = np.linspace(-3, 3, 500)
    pl.plot(x, np.exp(-0.5*x**2)/np.sqrt(2*np.pi))
    pl.savefig("test.png")
