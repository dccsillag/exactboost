import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.utils.eval_utils import eval_metrics, precision_at_k
from src.utils.general_utils import get_folder
from src.utils.model_utils import get_X_y


parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    "-d",
    default=None,
    help="Which dataset to use for training and prediction.",
)
parser.add_argument(
    "--model",
    "-m",
    default="exactboost_ks_single",
    help="Which model to use for training and prediction.",
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
    "--k",
    "-k",
    default=None,
    help="Precision of k topmost observations are to used when evaluating "
    "with precision at k metric (pak).",
)
parser.add_argument(
    "--beta",
    "-B",
    default=None,
    type=float,
    help="Alternative way of specifying k, where k = num_obs*beta;"
    "if k is specified, it has precedence.",
)
parser.add_argument(
    "--eval_fraction",
    default=0.5,
    type=float,
)
args = parser.parse_args()

CV_OR_VAL = "val" if args.validation else "cv"
DATASET = args.dataset
MODEL = args.model
TEST_SET = args.test_set
N_SPLITS = args.n_seeds if args.validation else args.n_folds

BETA = args.beta
K = args.k
EVAL_FRACTION = args.eval_fraction

FULL_DATA = f"{DATASET}_full_train+test"
X, y = get_X_y(FULL_DATA, seed=0)

if BETA is None and K is None:
    BETA = y.mean() * EVAL_FRACTION

res_auc = []
res_ks = []
res_pak = []

model_folder = get_folder(f"eval/boxplots-{CV_OR_VAL}/models")
preds_folder = get_folder(f"{model_folder}/{MODEL}/{DATASET}/preds/")

files = []

for i in range(N_SPLITS):
    files.append(preds_folder + f"{MODEL}-{CV_OR_VAL}-{TEST_SET}-{i}_{N_SPLITS}.h5")


if files == []:
    raise FileNotFoundError(
        "Could not find predictions for suggested model " f"(at {preds_folder})"
    )

for file in tqdm(files):
    df = pd.read_hdf(file)
    metrics = eval_metrics(df)
    res_auc.append(metrics["roc_auc"])
    res_ks.append(metrics["ks_statistic"])
    pak_val, pak_k = precision_at_k(
        df["target"].to_numpy(), df["preds"].to_numpy(), K, BETA
    )
    res_pak.append(pak_val / pak_k)

to_save = {
    "auc": res_auc,
    "ks": res_ks,
    "pak": res_pak,
}

output_folder = get_folder(f"eval/boxplots-{CV_OR_VAL}/{DATASET}")

for metric, res in to_save.items():
    output_file = f"{MODEL}-{CV_OR_VAL}-{TEST_SET}-{N_SPLITS}-{metric}.csv"
    np.savetxt(f"{output_folder}/{output_file}", res, delimiter=",")
    print(f"-saved: {output_folder}/{output_file}")
