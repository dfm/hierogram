import numpy as np
import matplotlib.pyplot as pl
from hierogram import Hierogram, GaussianProcess

np.random.seed(1234)
x = 5 + np.random.randn(100, 50, 1)
model = Hierogram(x, [np.linspace(2.5, 7.5, 10)])

n = 1000
hyper = np.empty((n, 3))
chain = np.empty((n, model.ndim))
prior = GaussianProcess([np.log(100), 2.6, 0.0], model.centers,
                        np.array([2.0, 0.5, 0.3]) / 2.4)
for i, x in enumerate(model.sample(prior, update_hyper=10)):
    chain[i] = x[0]
    hyper[i] = x[1]
    if i == n - 1:
        break

pl.plot(hyper[:, 0])
pl.savefig("hyper.png")

pl.clf()
x = np.array(zip(model.bins[0][:-1], model.bins[0][1:])).flatten()
for i in np.random.randint(n, size=50):
    y = np.array(zip(chain[i], chain[i])).flatten()
    pl.plot(x, np.exp(y), "k", alpha=0.5)
pl.savefig("test.png")
