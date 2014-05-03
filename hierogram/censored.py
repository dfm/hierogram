# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["CensoredHierogram", "PointwiseCensor"]

import numpy as np

from .utils import logsumexp
from .hierogram import Hierogram


class CensoredHierogram(Hierogram):

    def __init__(self, censor, *args, **kwargs):
        super(CensoredHierogram, self).__init__(*args, **kwargs)

        # Pre-integrate the censoring function onto the hierogram grid.
        self.censor_integral = censor.integrate_grid(self.bins).flatten()
        self.censor = censor

    def lnlike(self, theta):
        norm = np.exp(logsumexp(theta + self.censor_integral))
        grid = self.lnratefn(theta)
        vec = logsumexp(grid[self.inds] + self.censor(self.samples), axis=1)
        vec[~np.isfinite(vec)] = -np.inf
        return np.sum(vec) - norm


class PointwiseCensor(object):

    def __init__(self, samples, weights):
        self.samples = samples
        self.weights = weights
        self.bins = None
        self.grid = None

    def integrate_grid(self, bins):
        # Compute the empirical completeness.
        self.bins = bins
        numer, _ = np.histogramdd(self.samples, bins, weights=self.weights)
        denom, _ = np.histogramdd(self.samples, bins)
        m = denom > 0
        grid = -np.inf + np.zeros(numer.shape)
        grid[m] = np.log(numer[m]) - np.log(denom[m])

        # Pad the edges of the grid to deal with points outside the boundaries.
        shape = [len(b) - 1 for b in self.bins]
        self.grid = -np.inf + np.zeros([s+2 for s in shape])
        self.grid[[slice(1, -1)] * len(shape)] = grid

        # Compute the cell volumes.
        lnwidths = map(np.log, map(np.diff, bins))
        lnvolumes = reduce(np.add, np.ix_(*(lnwidths)))

        return grid + lnvolumes

    def __call__(self, x):
        assert self.bins is not None and self.grid is not None
        inds = [np.digitize(x[:, :, i].flatten(), b)
                .reshape(x[:, :, i].shape)
                for i, b in enumerate(self.bins)]
        return self.grid[inds]
