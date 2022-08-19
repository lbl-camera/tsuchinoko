import numpy as np


def transform(cc, k=100, tau=0.5):
    result = 0.5 + 0.5*np.tanh(k*(cc-tau))
    return result


def explore_target(N, k=100, tau=0.5):
    def acq_func(x, gp):

        mean = gp.posterior_mean(x)["f(x)"]
        cov = gp.posterior_covariance(x)["v(x)"]
        i = len(gp.points)
        if i <= N:
            return cov
        else:
            return transform(mean, k, tau) * cov
    return acq_func


explore_target_100 = explore_target(100, 1e-3, 255/2)
