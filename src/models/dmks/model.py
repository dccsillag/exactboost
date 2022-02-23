"""
DMKS: Directly Maximizing the Kolmogorov-Smirnov statistic.
Implementation of the method described by Fang and Chen (2019).
Some functions were inspired by the original R code, which was kindly shared by Fang.
"""

import numpy as np
import pandas as pd

from src.utils.eval_utils import ks


def imo(X, y, weight, beta_start, lamb=0, max_iter=500, tol=1e-06):
    """Iterative Marginal Optimization algorithm
       Returns the beta that maximizes KS(y, X*beta) and the corresponding KS

    Args:
        X: design matrix
        y: binary ground truth vector (default; non-default)
        weight: weighted penalty vector
        beta_start: initial estimated coefficient vector
        lamb: penalty factor
        max_iter: maximum number of iterations
        tol: tolerance for the stopping criterion
    """
    n0 = np.sum(y == 0)
    n1 = np.sum(y == 1)
    n, p = X.shape

    if p == 1:
        beta = 1
        ks_max = ks(pd.DataFrame({"target": y, "preds": X}))[0]
        return beta, ks_max

    beta = beta_start
    beta_prev = beta_start
    eps = np.finfo(float).eps
    ks_max, cutoff = ks(pd.DataFrame({"target": y, "preds": np.matmul(X, beta)}))

    itr = 1
    while itr < max_iter:
        for k in range(p):
            if beta[k] != 0:
                # in order to avoid division by zero, we add the machine epsilon to the denominator
                cuts = (
                    cutoff - np.matmul(np.delete(X, k, axis=1), np.delete(beta, k))
                ) / (X[:, k] + eps)
                idx = np.argsort(cuts)
                cuts = cuts[idx]
                y = y[idx]
                X = X[idx]

                ks_0 = np.empty(n)
                ks_1 = np.empty(n)

                ks_0[0] = (
                    np.sum(
                        cuts[0] * X[np.where(y == 0), k]
                        <= cuts[np.where(y == 0)] * X[np.where(y == 0), k]
                    )
                    / n0
                )
                ks_1[0] = (
                    np.sum(
                        cuts[0] * X[np.where(y == 1), k]
                        <= cuts[np.where(y == 1)] * X[np.where(y == 1), k]
                    )
                    / n1
                )

                ks_0, ks_1 = imo_rules(X, y, k, ks_0, ks_1, n0, n1, n)

                ks_tot = ks_0 - ks_1

                # penalize KS
                pks = ks_tot - (lamb * weight[k] * abs(cuts))
                pks0 = (
                    np.sum((cuts[np.where(y == 0)] * X[np.where(y == 0), k]) >= 0) / n0
                    - np.sum((cuts[np.where(y == 1)] * X[np.where(y == 1), k]) >= 0)
                    / n1
                )
                if np.max(pks) < pks0 and lamb > 0:
                    beta[k] = 0
                else:
                    # in case of a draw, select the highest cutoff
                    # we cannot use np.argmax(pks) because numpy's argmax returns only
                    # the first occurrence if there is a tie, which would correspond to
                    # the lowest cutoff instead of the highest
                    beta[k] = np.max(cuts[np.where(pks == np.max(pks))])

        if np.sum(beta ** 2) == 0:
            ks_max = ks(pd.DataFrame({"target": y, "preds": np.zeros(n)}))[0]
            break

        # normalize the coefficient vector so its euclidean norm equals one.
        beta = beta / np.sqrt(np.sum(beta ** 2))
        ks_max, cutoff = ks(pd.DataFrame({"target": y, "preds": np.matmul(X, beta)}))

        # stopping criterion
        if abs(1 - np.sum(beta_prev * beta)) < tol:
            break

        beta_prev = beta
        itr += 1

    return beta, ks_max


def imo_rules(X, y, k, ks_0, ks_1, n0, n1, n):
    """Auxiliary function, part of the the IMO algorithm
       Updates ks_0 and ks_1 according to the algorithm rules
    """
    for i in range(1, n):
        if y[i - 1] == 0 and y[i] == 0:  # case 2 and case 4
            if X[i - 1, k] > 0 and X[i, k] > 0:  # case 4
                ks_0[i] = ks_0[i - 1] - 1 / n0

            elif X[i - 1, k] < 0 and X[i, k] < 0:  # case 2
                ks_0[i] = ks_0[i - 1] + 1 / n0

            elif X[i - 1, k] * X[i, k] < 0:
                ks_0[i] = ks_0[i - 1]

            ks_1[i] = ks_1[i - 1]

        elif y[i - 1] == 0 and y[i] == 1:  # case 3 and case 5
            if X[i - 1, k] > 0:  # case 3
                ks_0[i] = ks_0[i - 1] - 1 / n0

            else:
                ks_0[i] = ks_0[i - 1]

            if X[i, k] < 0:  # case 5
                ks_1[i] = ks_1[i - 1] + 1 / n1

            else:
                ks_1[i] = ks_1[i - 1]

        elif y[i - 1] == 1 and y[i] == 0:  # case 1 and case 7
            if X[i, k] < 0:  # case 1
                ks_0[i] = ks_0[i - 1] + 1 / n0

            else:
                ks_0[i] = ks_0[i - 1]

            if X[i - 1, k] > 0:  # case 7
                ks_1[i] = ks_1[i - 1] - 1 / n1

            else:
                ks_1[i] = ks_1[i - 1]

        elif y[i - 1] == 1 and y[i] == 1:  # case 6 and case 8
            if X[i - 1, k] > 0 and X[i, k] > 0:  # case 8
                ks_1[i] = ks_1[i - 1] - 1 / n1

            elif X[i - 1, k] < 0 and X[i, k] < 0:  # case 6
                ks_1[i] = ks_1[i - 1] + 1 / n1

            elif X[i - 1, k] * X[i, k] < 0:
                ks_1[i] = ks_1[i - 1]

            ks_0[i] = ks_0[i - 1]

    return ks_0, ks_1


def pava(val, weight):
    """Pool-Adjacent-Violators Algorithm
       Returns the monotonically non-decreasing sequence that best fits the input data

    Args:
        val: input values to be approximated
        weight: weighting factor
    """
    n = len(val)
    y = {0: val[0]}
    w = {0: weight[0]}
    S = {0: 0, 1: 1}
    j = 0
    for i in range(1, n):
        j += 1
        y[j] = val[i]
        w[j] = weight[i]
        while j > 0 and y[j] < y[j - 1]:
            y[j - 1] = (w[j] * y[j] + w[j - 1] * y[j - 1]) / (w[j] + w[j - 1])
            w[j - 1] += w[j]
            j -= 1
        S[j + 1] = i + 1
    y_hat = np.empty(n)
    for k in range(j + 1):
        for l in range(S[k], S[k + 1]):
            y_hat[l] = y[k]
    return y_hat


def gauss_kern(t):
    """Gaussian Kernel function"""
    return np.exp(-(t ** 2) / 2) / np.sqrt(2 * np.pi)


def pestimate(score, y, out):
    """Estimates the default probability

    Args:
        score: in-sample score
        y: binary ground truth vector (default; non-default)
        out: out-of-sample score
    """
    n = len(y)
    xs = np.sort(score)
    ys = pava(y[np.argsort(score)], np.ones(n))
    lo = len(out)
    rh = np.empty(lo)
    fh = np.empty(lo)
    yh = np.empty(lo)
    h = n ** (-1 / 5)
    for j in range(lo):
        rh[j] = np.sum(gauss_kern((out[j] - xs) / h) * ys)
        fh[j] = np.sum(gauss_kern((out[j] - xs) / h))
        # "Suppose the denominator equals zero. In this case, the numerator is also
        # equal to zero, so we set the estimate to 0." -- Wolfgang Härdle on the
        # Nadaraya-Watson estimator (Smoothing Techniques: With Implementation in S)
        if fh[j] == 0:
            yh[j] = 0
        else:
            yh[j] = rh[j] / fh[j]
    return yh
