# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["GaussianProcess"]

import numpy as np

try:
    from scipy.linalg import cholesky, cho_solve
except ImportError:
    cholesky = None


class GaussianProcess(object):

    def __init__(self, params, centers, proposal=None, randomize=False,
                 eps=1e-8):
        if cholesky is None:
            raise ImportError("Install scipy to use a GP prior")

        self.eps = eps
        self.proposal = proposal
        self.randomize = randomize

        if len(centers) > 1:
            coords = np.meshgrid(*centers, indexing="ij")
        else:
            coords = centers[0]
        coords = np.vstack([c.flatten() for c in coords]).T
        self.dvec = np.array([reduce(np.add, np.ix_(c, -c)) for c in coords])
        self.dvec = self.dvec ** 2
        self.ndim = self.dvec.shape[2] + 2
        self.params = params

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, v):
        self._params = np.array(v)
        chi2 = np.sum(self.dvec/np.exp(self._params[2:self.ndim]), axis=0)
        K = np.exp(self._params[1] - 0.5 * chi2)
        K[np.diag_indices_from(K)] += self.eps
        self.factor = cholesky(K, lower=True, overwrite_a=True)
        self.logdet = np.sum(2*np.log(np.diag(self.factor)))

    def __call__(self, theta):
        y = theta - self.params[0]
        lp = -0.5 * (np.dot(y, cho_solve((self.factor, True), y))+self.logdet)
        return lp if np.isfinite(lp) else -np.inf

    def _metropolis_step(self, theta):
        assert self.proposal is not None, \
            "You need to choose a proposal"
        if self.randomize:
            step = self.proposal * np.random.rand(len(self.proposal))
        else:
            step = self.proposal

        # Compute the initial probabilities.
        lp0 = self(theta)
        q = self.params + step * np.random.randn(len(self.params))

        # Cache the factor and logdet for the case where we don't accept the
        # update.
        old_params = self.params
        old_factor, old_logdet = self.factor, self.logdet

        # Compute the model at the proposed position.
        self.params = q
        lp1 = self(theta)

        if np.isfinite(lp1) and np.exp(lp1 - lp0) >= np.random.rand():
            return q, lp1, True

        # Revert to the previous value.
        self._params = old_params
        self.factor = old_factor
        self.logdet = old_logdet
        return self.params, lp0, False
