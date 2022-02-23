from math import ceil

import numpy as np
import pandas as pd
from scipy.stats import binned_statistic
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    make_scorer,
    matthews_corrcoef,
    roc_auc_score,
)


def ks(df, target="target", preds="preds"):
    """Compute the Kolmogorov-Smirnov statistic and its corresponding cut-off score
    from a Pandas DataFrame with preds and target columns.

    Args:
        df (pd.DataFrame): DataFrame containing target and preds columns.
        target (str): Column name with ground truth.
        preds (str): Column name with predictions.

    Returns:
        tuple: (Kolmogorov-Smirnov statistic, cut-off score)
    """
    y_pred = df[preds].to_numpy()
    y_truth = df[target].to_numpy()
    ks, threshold = ks_np(y_truth, y_pred, return_threshold=True)
    return ks, threshold


def ks_np(y_truth: np.ndarray, y_pred: np.ndarray, threshold=None, return_threshold=False):
    """Compute the Kolmogorov-Smirnov statistic and its corresponding cut-off score
    from a Pandas DataFrame with preds and target columns.

    Obs.: does the same thing as `ks` above, except it uses NumPy arrays instead of
    Pandas DataFrames.

    Args:
        y_pred (array-like): Array containing the predicted scores.
        y_truth (array-like): Array containing the true labels.
        threshold (float): If given, calculate KS at this point instead of estimating
          the optimal cutoff.

    Returns:
        tuple: (Kolmogorov-Smirnov statistic, cut-off score)
    """
    s0 = 1 / np.sum(y_truth == 0)
    s1 = 1 / np.sum(y_truth == 1)
    unique_scores, obs_by_scores = np.unique(y_pred, return_inverse=True)
    variation = np.where(y_truth, -s1, s0)
    ks_values = np.zeros(len(unique_scores))
    for i in range(len(y_truth)):
        ks_values[obs_by_scores[i]] = ks_values[obs_by_scores[i]] + variation[i]
    ks_values = np.cumsum(ks_values)
    if threshold:
        if threshold not in unique_scores:
            unique_scores = np.append(unique_scores, threshold)
            unique_scores = np.sort(unique_scores)
            ks = float(ks_values[np.argwhere(unique_scores == threshold)[0] - 1])
        else:
            ks = float(ks_values[np.argwhere(unique_scores == threshold)[0]])
    else:
        threshold = unique_scores[np.argmax(ks_values)]
        ks = np.max(ks_values)

    if return_threshold:
        return ks, threshold
    else:
        return ks


def ks_np_theta(y_truth: np.ndarray, y_pred: np.ndarray, margin_theta: float, threshold=None, return_threshold=False):
    y_pred_theta = y_pred - y_truth * margin_theta

    ks, threshold = ks_np(y_truth, y_pred_theta, return_threshold=True)

    if return_threshold:
        return ks, threshold
    else:
        return ks


def information_value(df, target="target", preds="preds", n_bins=10):
    """Compute the information value statistic from a Pandas DataFrame with preds and
    and target columns.

    Args:
        df (pd.DataFrame): DataFrame containing target and preds columns.
        target (str): Column name with ground truth.
        preds (str): Column name with predictions.
        n_bins (int): Number of bins to divide the data.
    """
    y = df[target].to_numpy()
    preds = df[preds].to_numpy()

    bin_mean = binned_statistic(preds, y, bins=n_bins, statistic="mean")[0]
    bin_count = binned_statistic(preds, y, bins=n_bins, statistic="count")[0]

    ones_per_bin = bin_mean * bin_count
    zeros_per_bin = bin_count - (ones_per_bin)

    # a lack of observation (nan) is equivalent in this context to zero observations
    ones_per_bin = np.nan_to_num(ones_per_bin)
    zeros_per_bin = np.nan_to_num(zeros_per_bin)

    total_ones = np.sum(y == 1)
    total_zeros = np.sum(y == 0)

    ones_pct = ones_per_bin / total_ones
    zeros_pct = zeros_per_bin / total_zeros

    # the machine epsilon will be added to avoid division by zero and log(0)
    eps = np.finfo(float).eps

    # calculate information value
    iv = np.sum((ones_pct - zeros_pct) * np.log((ones_pct + eps) / (zeros_pct + eps)))

    return iv


def cohens_d(df, target="target", preds="preds"):
    """Compute Cohen's d from a Pandas DataFrame with preds and and target columns.

    Args:
        df (pd.DataFrame): DataFrame containing target and preds columns.
        target (str): Column name with ground truth.
        preds (str): Column name with predictions.
    """
    y = df[target].to_numpy()
    preds = df[preds].to_numpy()

    mu_ones = np.mean(preds[y == 1])
    mu_zeros = np.mean(preds[y == 0])

    n_ones = np.sum(y == 1)
    n_zeros = np.sum(y == 0)

    sigma2_ones = np.std(preds[y == 1], ddof=1) ** 2
    sigma2_zeros = np.std(preds[y == 0], ddof=1) ** 2

    sigma = np.sqrt(
        ((n_ones - 1) * sigma2_ones + (n_zeros - 1) * sigma2_zeros)
        / (n_ones + n_zeros - 2)
    )

    return (mu_ones - mu_zeros) / sigma


def precision_at_k(y_truth, y_pred, k=None, beta=0.2, top=True):
    """Compute the precision at k (p@k) performance metric.
    Args:
        y_pred (array-like): Array containing the predicted scores.
        y_truth (array-like): Array containing the true labels.
        k (int): Number of scores to be used.
        beta (float): Proportion of scores used to define k if no k is given.
        top (boolean): Whether to use the top-k or bottom-k scores.

    Returns:
        tuple: (Precision at k, k)
    """
    assert len(y_pred) == len(y_truth), "y_pred and y_truth must have the same size"
    assert beta > 0 and beta <= 1, "beta must be positive and less than one"

    if k is None:
        k = ceil(beta * len(y_pred))

    assert k >= 1 and k <= len(
        y_pred
    ), "k must be at least 1 and at most the total number of scores"

    idx = np.lexsort((1 - y_truth, y_pred))

    if top:
        idx = idx[-k:]
    else:
        idx = idx[:k]

    return (np.sum(y_truth[idx]), k)


def eval_metrics(
    df,
    target="target",
    preds="preds",
    metrics=[
        "ks_point",
        "ks_statistic",
        "roc_auc",
        "accuracy",
        "information_value",
        "cohens_d",
        "mcc",
        "pak",
    ],
    k=None,
    beta=None,
    print_conf_matrix=False,
):
    """Create report with main evaluation metrics for credit scoring.

    Args:
        df (pd.DataFrame): DataFrame containing target and preds columns.
        target (str): Column name with ground truth.
        preds (str): Column name with predictions.
        metrics (list): List with metrics to include in report.

    Returns:
        pd.DataFrame: DataFrame with evaluated metrics.
    """
    # ensure binary target values are integers
    df[target] = df[target].astype(int)

    # calculate ks_statistic and ks_point
    ks_statistic, ks_point = ks(df, target, preds)

    # set report
    report = {}
    if "ks_statistic" in metrics:
        report["ks_statistic"] = ks_statistic
    if "ks_point" in metrics:
        report["ks_point"] = ks_point
    if "roc_auc" in metrics:
        roc_auc = roc_auc_score(y_true=df[target], y_score=df[preds])
        report["roc_auc"] = roc_auc
    if "pak" in metrics and (k is not None or beta is not None):
        pak, k = precision_at_k(df[target].to_numpy(), df[preds].to_numpy(), k=k, beta=beta)
        report["pak"] = pak / k
    if "accuracy" in metrics:
        report["accuracy"] = accuracy_score(df[target], df[preds].round())
    if "information_value" in metrics:
        report["information_value"] = information_value(df, target, preds)
    if "cohens_d" in metrics:
        report["cohens_d"] = cohens_d(df, target, preds)
    if "mcc" in metrics:
        report["mcc"] = matthews_corrcoef(df[target], df[preds].round())
    if print_conf_matrix is True:
        print_confusion_matrix(df[target], df[preds].round())

    return pd.Series(report)


def print_confusion_matrix(y_true, y_hat):
    """Print a confusion matrix from real observations and corresponding predictions"""
    conf_matrix = pd.DataFrame(confusion_matrix(y_true, y_hat))
    print("\nConfusion matrix")
    print("Rows: truth\nColumns: prediction\n")
    print(conf_matrix, "\n")

