import argparse

import numpy as np
from joblib import Parallel, delayed
from sklearn.model_selection import KFold
from tqdm import tqdm

from src.utils.model_utils import get_X_y

from src.eval.boxplots.utils import combine_predictions


parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    "-d",
    default=None,
    help="Which dataset to use for training and prediction.",
)
parser.add_argument(
    "--models",
    "-m",
    nargs="+",
    default=["adaboost", "knn", "logistic", "neural_network", "random_forest", "xgboost"],
    help="Models whose predictions are to be combined.",
)
parser.add_argument(
    "--test_set",
    "-ts",
    default="test",
    choices=["ens", "ensemble", "test"],
    help="Whether to create test set from ensemble data "
    "(i.e., as subset of train data) or from test data.",
)
parser.add_argument(
    "--validation",
    "-val",
    action="store_true",
    help="If True, repeatedly subsample data to generate boxplot train and test folds; "
    "if False, use cross-validation (i.e., use independent folds for train and test).",
)
parser.add_argument(
    "--n_jobs",
    "-j",
    type=int,
    default=1,
    help="How many jobs to launch in parallel, one for each CV fold.",
)
parser.add_argument(
    "--n_folds",
    "-f",
    type=int,
    default=5,
    help="Number of folds to use in cross-validation.",
)
parser.add_argument(
    "--n_seeds",
    "-s",
    type=int,
    default=10,
    help="Number of seeds for resampling; "
    "only used in validation mode (i.e., CV_OR_VAL == 'VAL').",
)
parser.add_argument(
    "--preds_only",
    "-p",
    action="store_true",
    help="If True, combine only predictions; if False, include original features as well.",
)

args = parser.parse_args()

CV_OR_VAL = "val" if args.validation else "cv"
DATASET = args.dataset
MODELS = args.models
N_JOBS = args.n_jobs
TEST_SET = args.test_set
N_SPLITS = args.n_seeds if args.validation else args.n_folds
PREDS_ONLY = args.preds_only

MODEL_PARAMS = [
    CV_OR_VAL,
    DATASET,
    MODELS,
    TEST_SET,
    N_SPLITS,
    PREDS_ONLY,
]

FULL_DATA = f"{args.dataset}_full_train+test"
X, y = get_X_y(FULL_DATA, seed=0)

model_inputs = []

if CV_OR_VAL == "cv":
    # Generate folds
    for i, (train_index, test_index) in enumerate(
        KFold(n_splits=N_SPLITS, random_state=0, shuffle=True).split(X)
    ):
        if TEST_SET.startswith("ens"):
            # Split train fold to generate test set
            rnd = np.random.RandomState(i)
            size_of_ens = round((1 / N_SPLITS) * len(train_index))
            ens_index = rnd.choice(train_index, size=size_of_ens, replace=False)
            train_index = np.setdiff1d(train_index, test_index)
            model_inputs.append((i, train_index, ens_index, *MODEL_PARAMS))
        else:
            # Use test fold as test set
            model_inputs.append((i, train_index, test_index, *MODEL_PARAMS))
elif CV_OR_VAL == "val":
    n = X.shape[0]
    size_of_test = round((1 / 5) * n)
    # Generate random subsample of data
    for i in range(N_SPLITS):
        rnd = np.random.RandomState(i)
        test_index = rnd.choice(range(n), size=size_of_test, replace=False)
        train_index = np.setdiff1d(range(n), test_index)

        if TEST_SET.startswith("ens"):
            # Create ensemble fold from train fold and use it as test
            size_of_ens = round((1 / 5) * len(train_index))
            ens_index = rnd.choice(train_index, size=size_of_ens, replace=False)
            train_index = np.setdiff1d(train_index, ens_index)
            model_inputs.append((i, train_index, ens_index, *MODEL_PARAMS))
        else:
            # Use test fold as data
            model_inputs.append((i, train_index, test_index, *MODEL_PARAMS))

Parallel(n_jobs=N_JOBS)(
    delayed(combine_predictions)(*model_input) for model_input in tqdm(model_inputs)
)
