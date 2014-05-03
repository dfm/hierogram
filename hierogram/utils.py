# -*- coding: utf-8 -*-

__all__ = ["logsumexp"]

import numpy as np

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
