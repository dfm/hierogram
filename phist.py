#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module provides an interface for finding the histogram of a set of noisy
observations.

"""

from __future__ import division, print_function

__all__ = ["PHist", "phist"]

__version__ = "0.0.1"

__author__ = "Dan Foreman-Mackey (danfm@nyu.edu)"
__contributors__ = []

__copyright__ = "Copyright 2014 Daniel Foreman-Mackey"
__license__ = "MIT"

import logging
import numpy as np

try:
    import scipy.optimize as op
except ImportError:
    op = None

try:
    from scipy.misc import logsumexp
except ImportError:
    def logsumexp(a, axis=None, b=None):
        """
        This function is taken directly from scipy in order to support older
        versions.

        """
        a = np.asarray(a)
        if axis is None:
            a = a.ravel()
        else:
            a = np.rollaxis(a, axis)
        a_max = a.max(axis=0)
        if b is not None:
            b = np.asarray(b)
            if axis is None:
                b = b.ravel()
            else:
                b = np.rollaxis(b, axis)
            out = np.log(sum(b * np.exp(a - a_max), axis=0))
        else:
            out = np.log(sum(np.exp(a - a_max), axis=0))
        out += a_max
        return out


class PHist(object):
    """
    A probabilistic histogram.

    :param x: ``(K, N)``
        A list of ``N`` posterior samples for measurements of ``K`` examples.

    :param bins:
        Either the integer number of bins or an array of bin edges.

    :param lnpriors: ``(K, N)`` or ``None``
        If not ``None``, this should be an array with the log prior evaluated
        at the samples given in ``x``.

    """

    def __init__(self, x, bins=10, lnpriors=None):
        # Make sure that the samples have the correct shape.
        x = np.atleast_2d(x)

        # Interpret the bin specs.
        try:
            len(bins)
        except TypeError:
            bins = np.linspace(x.min(), x.max(), bins)
        self.bins = np.array(bins)

        # Pre-compute the bin indices of the samples.
        self.inds = np.digitize(x.flatten(), bins).reshape(x.shape)

        # Generate a prior array if one wasn't provided.
        if lnpriors is None:
            self.lnpriors = np.zeros_like(x)
        else:
            self.lnpriors = np.array(lnpriors)

        # Make an initial guess at the histogram values.
        self.theta = np.array(np.histogram(np.mean(x, axis=1), bins)[0],
                              dtype=float)
        m = self.theta > 0
        self.theta[m] = np.log(self.theta[m])
        self.theta[~m] = -np.inf

        # Normalize the bin values to sum to one.
        self.theta -= logsumexp(self.theta)

    @property
    def density(self):
        """
        The bin heights of the histogram normalized so that the integral over
        all the bins is one.

        """
        norm = logsumexp(self.theta + np.log(self.bins[1:] - self.bins[:-1]))
        return np.exp(self.theta - norm)

    def lnlike(self, p):
        """
        Compute the marginalized log-likelihood of the samples given a set of
        proposed bin heights.

        :param p:
            A list of bin heights. This must have length ``nbins-1``.

        """
        theta = np.empty(len(p)+2)
        theta[1:-1] = p - logsumexp(p)
        theta[0] = -np.inf
        theta[-1] = -np.inf
        return np.sum(logsumexp(theta[self.inds] - self.lnpriors, axis=1))

    def nll(self, p):
        """
        Compute the negative marginalized log-likelihood of the data. This is
        legitimately just the negative of the ``lnlike`` method.

        """
        return -self.lnlike(p)

    def optimize(self, **kwargs):
        """
        Use ``scipy.optimize.minimize`` to find the maximum marginalized
        likelihood histogram. In particular, this function uses the bounded
        ``L-BFGS-B`` method.

        :param ** kwargs:
            Keyword arguments passed directly to the ``minimize`` call. Any
            values for ``method`` or ``bounds`` will be ignored.

        :returns result:
            The results dictionary from the minimize call.

        """
        if op is None:
            raise ImportError("Install scipy to optimize your phist.")

        # Try and support older versions of scipy that don't have the
        # "minimize" API.
        if hasattr(op, "minimize"):
            kwargs["method"] = "L-BFGS-B"
            minimize = op.minimize
        else:
            logging.warn("Using legacy optimization interface. Consider "
                         "upgrading scipy.")
            kwargs.pop("method", None)
            minimize = op.fmin_l_bfgs_b

        # Ignore any provided bounds.
        kwargs["bounds"] = [(None, 0)]*len(self.theta)

        # Run the optimization to get the maximum likelihood histogram.
        p0 = self.theta - logsumexp(self.theta)
        results = minimize(self.nll, p0, **kwargs)

        # Save the results of the optimization.
        self.theta = np.array(results.x)

        return results


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
    p = PHist(x, bins=bins, lnpriors=lnpriors)
    p.optimize()
    return p.density, p.bins


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
