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
    from scipy.linalg import cho_factor, cho_solve
except ImportError:
    cho_factor = None

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

    def __init__(self, x, bins=10, rng=None, lnpriors=None):
        # Make sure that the samples have the correct shape.
        x = np.atleast_2d(x)

        # Interpret the bin specs.
        try:
            len(bins)
        except TypeError:
            if rng is None:
                bins = np.linspace(x.min(), x.max(), bins + 1)
            else:
                bins = np.linspace(rng[0], rng[1], bins + 1)
        self.bins = np.array(bins)
        self.bin_widths = np.diff(self.bins)
        self.ln_bin_widths = np.log(self.bin_widths)
        self.bin_centers = self.bins[:-1] + 0.5*self.bin_widths

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
        return np.exp(self._get_log_density(self.theta))

    def _get_log_density(self, values):
        norm = logsumexp(values + self.ln_bin_widths)
        return values - norm

    def lnlike(self, p):
        """
        Compute the marginalized log-likelihood of the samples given a set of
        proposed bin heights.

        :param p:
            A list of bin heights. This must have length ``nbins-1``.

        """
        theta = np.empty(len(p)+2)
        theta[1:-1] = self._get_log_density(p)
        theta[0] = -np.inf
        theta[-1] = -np.inf
        return np.sum(logsumexp(theta[self.inds] - self.lnpriors, axis=1))

    def nll(self, p):
        """
        Compute the negative marginalized log-likelihood of the data. This is
        legitimately just the negative of the ``lnlike`` method.

        """
        return -self.lnlike(p)

    def optimize(self, minval=-16.0, **kwargs):
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
            kwargs["approx_grad"] = True
            minimize = op.fmin_l_bfgs_b

        # Ignore any provided bounds.
        kwargs["bounds"] = [(None, 0)]*len(self.theta)

        # Run the optimization to get the maximum likelihood histogram.
        p0 = np.array(self.theta)
        m = np.isfinite(p0)
        p0[~m] = np.min(p0[m])
        p0 -= logsumexp(p0)
        results = minimize(self.nll, p0, **kwargs)

        # Save the results of the optimization.
        self.theta = np.array(results.x)

        return results

    def _ess_step(self, f0, ll0, cov, tp=2*np.pi):
        D = len(f0)
        nu = np.dot(cov, np.random.randn(D))
        lny = ll0 + np.log(np.random.rand())
        th = tp*np.random.rand()
        thmn, thmx = th-tp, th
        i = 0
        while True:
            fp = f0*np.cos(th) + nu*np.sin(th)
            ll = self.lnlike(fp)
            i += 1
            if ll > lny:
                return fp, ll
            if th < 0:
                thmn = th
            else:
                thmx = th
            th = np.random.uniform(thmn, thmx)

    def _lnprior_eval(self, pars, heights):
        a, s = np.exp(2*pars)
        cov = a * np.exp(-0.5 * (self.bin_centers[:, None] -
                                 self.bin_centers[None, :])**2 / s)
        cov[np.diag_indices_from(cov)] += 1e-8
        try:
            factor, flag = cho_factor(cov)
        except:
            return -np.inf, cov
        logdet = np.sum(2*np.log(np.diag(factor)))
        return -0.5 * (np.dot(heights, cho_solve((factor, flag), heights))
                       + logdet), cov

    def _metropolis_step(self, pars, heights):
        lp0, cov0 = self._lnprior_eval(pars, heights)
        q = pars + (np.random.rand(2))*np.random.randn(2)
        lp1, cov1 = self._lnprior_eval(q, heights)
        diff = lp1 - lp0
        if diff >= 0.0 or np.exp(diff) >= np.random.rand():
            return q, cov1
        return pars, cov0

    def sample(self, nsamples=1000, minval=-10.):
        if cho_factor is None:
            raise ImportError("Install scipy to sample")

        x = self.bins[:-1] + 0.5 * np.diff(self.bins)
        cov = 26**2 * np.exp(-0.5 * ((x[:, None] - x[None, :])/2.0)**2)
        cov[np.diag_indices_from(cov)] += 1e-10
        pars = np.log([0.1, 2])
        print(np.exp(pars))

        theta = np.dot(cov, np.random.randn(len(x)))
        ll = self.lnlike(theta)
        samples = np.empty((nsamples, len(x)))
        hypers = np.empty((nsamples, len(pars)))

        for i in range(nsamples):
            theta, ll = self._ess_step(theta, ll, cov)
            samples[i, :] = theta
            pars, cov = self._metropolis_step(pars, theta)
            hypers[i, :] = pars

        print(np.median(np.exp(hypers), axis=0))

        samples = map(self._get_log_density, samples[-500:, :])

        # Find the quantiles.
        v = np.array([quantile(t, [0.16, 0.5, 0.84])
                      for t in np.exp(samples).T]).T

        # Save the mean result and return the stats.
        self.theta = v[1]

        return v


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
    print(p.optimize())
    return p.density, p.bins


def quantile(x, q, weights=None):
    """
    Like numpy.percentile, but:

    * Values of q are quantiles [0., 1.] rather than percentiles [0., 100.]
    * scalar q not supported (q must be iterable)
    * optional weights on x

    """
    if weights is None:
        return np.percentile(x, [100. * qi for qi in q])
    idx = np.argsort(x)
    cdf = np.add.accumulate(weights[idx])
    cdf /= cdf[-1]
    return np.interp(q, cdf, x[idx]).tolist()


if __name__ == "__main__":
    import matplotlib.pyplot as pl

    np.random.seed(1234)

    K, N = 50, 300
    means = 10 + np.random.randn(K)
    err = 0.1 * np.exp(means)
    means = np.log(np.abs(np.exp(means) + err*np.random.randn(K)))
    samples = np.log(np.abs(np.exp(means[:, None])
                            + err[:, None]*np.random.randn(K, N)))
    print(np.any(~np.isfinite(samples)))

    p = PHist(samples, bins=40, rng=[5, 15])
    v_m, v, v_p = p.sample()
    bins = p.bins

    bin_edges = np.array(zip(bins[:-1], bins[1:])).flatten()
    mean = np.array(zip(v, v)).flatten()
    minus = np.array(zip(v_m, v_m)).flatten()
    plus = np.array(zip(v_p, v_p)).flatten()

    x = np.linspace(5, 15, 500)
    pl.plot(x, np.exp(-0.5*(x-10)**2)/np.sqrt(2*np.pi))

    pl.plot(bin_edges, mean, color="k")
    pl.plot(bin_edges, plus, color="k")
    pl.fill_between(bin_edges, plus, minus, color="k", alpha=0.3)

    pl.hist(means, bins, normed=True, histtype="step", color="r", lw=2)
    pl.savefig("test.png")
