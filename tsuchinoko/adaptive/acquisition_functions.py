from functools import partial

import numpy as np


def transform(cc, k=100, tau=0.5):
    result = 0.5 + 0.5*np.tanh(k*(cc-tau))
    return result


def explore_target(x, gp, N, k=100, tau=0.5):
    mean = gp.posterior_mean(x)["f(x)"]
    cov = gp.posterior_covariance(x)["v(x)"]
    i = len(gp.points)
    if i <= N:
        return cov
    else:
        return transform(mean, k, tau) * cov


explore_target_100 = partial(explore_target, N=100, k=1e-3, tau=255/2)

explore_target_100 = explore_target(100, 1e-3, 255/2)
