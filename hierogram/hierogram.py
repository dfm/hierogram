# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["Hierogram"]

import numpy as np
from itertools import izip
try:
    import scipy.optimize as op
except ImportError:
    op = None

from .utils import logsumexp


class Hierogram(object):
    """
    A probabilistic histogram.

    :param x: ``(K, N, D)`` or ``(K, N)``
        A list of ``N`` posterior samples for measurements of ``K`` examples
        in ``D`` dimensions.

    :param bins:
        The bin specification:
        * A sequence of arrays describing the bin edges along each dimension.
        * The number of bins for each dimension ``(nx, ny, ... =bins)``
        * The number of bins for all dimensions ``(nx=ny=...=bins)``.

    :param lnweights:
        If not ``None``, this should be an array with the log prior evaluated
        at the samples given in ``x``.

    """

    def __init__(self, samples, bins=10, weights=None, lnweights=None,
                 range=None):
        # Parse the input list of samples.
        self.samples = np.atleast_3d(samples)
        assert weights is None or lnweights is None, \
            "You should only specify a weights matrix *or* a lnweights matrix"
        if lnweights is None:
            self.lnweights = np.zeros_like(self.samples, dtype=float)
        else:
            self.lnweights = np.atleast_3d(lnweights)
        assert self.lnweights.shape == self.samples.shape, \
            "Dimension mismatch between sample list and weights"

        # Get the dimensions.
        K, N, D = self.samples.shape
        if D == 1:
            try:
                len(bins)
            except TypeError:
                pass
            else:
                bins = np.atleast_2d(bins)

        # Compute the bins using numpy's histogram function for consistency.
        _, self.bins = np.histogramdd(self.samples.reshape((-1, D)),
                                      bins=bins, range=range)
        self.shape = [len(b) - 1 for b in self.bins]
        self._grid = -np.inf + np.zeros([s+2 for s in self.shape])
        self._center = [slice(1, -1)] * D

        # Digitize the samples in advance.
        self.inds = [np.digitize(self.samples[:, :, i].flatten(), b)
                     .reshape((K, N))
                     for i, b in enumerate(self.bins)]

        # Compute the bin volumes.
        widths = map(np.diff, self.bins)
        self.lnwidths = map(np.log, widths)
        self.centers = [b[:-1] + 0.5*w for b, w in izip(self.bins, widths)]
        self.lnvolumes = reduce(np.add, np.ix_(*(self.lnwidths)))
        self.lnvolumes = self.lnvolumes.flatten()
        self.ndim = len(self.lnvolumes)

    def initial(self):
        w = np.exp(self.lnweights)
        x = np.sum(self.samples * w, axis=1) / np.sum(w, axis=1)
        return np.log(np.histogramdd(x, self.bins)[0] + 1).flatten()

    def lnratefn(self, theta):
        self._grid[self._center] = theta.reshape(self.shape)
        return self._grid

    def lnlike(self, theta):
        grid = self.lnratefn(theta)
        norm = np.exp(logsumexp(theta + self.lnvolumes))
        vec = logsumexp(grid[self.inds], axis=1)
        vec[~np.isfinite(vec)] = -np.inf
        return np.sum(vec) - norm

    def optimize(self, verbose=False, lnpriorfn=None, **kwargs):
        if op is None or not hasattr(op, "minimize"):
            raise ImportError("Install or upgrade scipy to optimize the "
                              "Hierogram")

        # Run the optimization.
        kwargs["method"] = "L-BFGS-B"
        if lnpriorfn is None:
            nll = lambda p: -self.lnlike(p)
        else:
            nll = lambda p: -(self.lnlike(p) + lnpriorfn(p))
        results = op.minimize(nll, self.initial(), **kwargs)

        # Print the results and return.
        if verbose:
            print(results)
        return results.x

    def _ess_step(self, factor, f0, ll0):
        D = len(f0)
        nu = np.dot(factor, np.random.randn(D))
        lny = ll0 + np.log(np.random.rand())
        th = 2*np.pi*np.random.rand()
        thmn, thmx = th-2*np.pi, th
        while True:
            fp = f0*np.cos(th) + nu*np.sin(th)
            ll = self.lnlike(fp)
            if ll > lny:
                return fp, ll
            if th < 0:
                thmn = th
            else:
                thmx = th
            th = np.random.uniform(thmn, thmx)

    def sample(self, lnprior, update_hyper=0):
        hyper = lnprior.params
        p = self.initial()
        ll = self.lnlike(p)
        lp = lnprior(p)
        count = 0
        accepted, total = 1, 1
        while True:
            p, ll = self._ess_step(lnprior.factor, p, ll)
            count += 1
            if update_hyper > 0 and count % update_hyper == 0:
                hyper, lp, a = lnprior._metropolis_step(p)
                accepted += a
                total += 1
            yield p, hyper, lp + ll, float(accepted) / total
