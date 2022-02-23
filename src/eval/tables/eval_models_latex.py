from argparse import ArgumentParser
import os

import numpy as np
import pandas as pd

from src.utils.general_utils import get_folder

pd.options.mode.chained_assignment = None

parser = ArgumentParser()
parser.add_argument(
    "--datasets",
    "-d",
    nargs="+",
    default=[
        "a1a",
        "australian",
        "banknote",
        "breast-cancer",
        "cod-rna",
        "colon-cancer",
        "covtype",
        "cskaggle",
        "diabetes",
        "fourclass",
        "german",
        "gisette",
        "gmsc",
        "heart",
        "housing",
        "ijcnn1",
        "ionosphere",
        "liver-disorders",
        "madelon",
        "mammography",
        "mq2008",
        "oil-spill",
        "phishing",
        "phoneme",
        "skin-nonskin",
        "sonar",
        "splice",
        "svmguide1",
        "svmguide3",
        "taiwan",
        "w1a",
    ],
    help="Which dataset to use",
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
    help="If True, repeatedly subsample data to generate boxplot train and test folds;"
    "if False, use cross-validation (i.e., use independent folds for train and test)."
,
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
    "--metric",
    "-m",
    default="ks",
    help="Metric used to evaluate models performance"
)
parser.add_argument(
    "--ensemblers",
    "-ens",
    action="store_true",
    help="If True, evaluate ensemblers instead of estimators"
)
parser.add_argument(
    "--filename",
    default=None,
    help="If set, overrides filename used when exporting table"
)
args = parser.parse_args()

METRIC = args.metric
CV_OR_VAL = "val" if args.validation else "cv"
MODELS_TYPE = "ensemblers" if args.ensemblers else "estimators"
FILENAME = args.filename

if METRIC == "auc" or METRIC == "ks" or METRIC == "pak":
    EXACTBOOST = [f"exactboost_{METRIC}"]
else:
    # If metric is not auc, ks or pak, present all exactboost models for comparison
    EXACTBOOST = ["exactboost_auc", "exactboost_ks", "exactboost_pak"]

if MODELS_TYPE == "estimators":
    MODELS = [
        "adaboost",
        "knn",
        "logistic",
        "neural_network",
        "random_forest",
        "xgboost",
        *EXACTBOOST,
    ]
    BENCHMARK_MODEL = None

    if METRIC == "auc":
        BENCHMARK_MODEL = "rankboost"
    elif METRIC == "ks":
        BENCHMARK_MODEL = "dmks"
    elif METRIC == "pak":
        BENCHMARK_MODEL = "top_push"

    if BENCHMARK_MODEL:
        MODELS.append(BENCHMARK_MODEL)

elif MODELS_TYPE == "ensemblers":
    MODELS = [
        "adaboost_ensembler",
        "knn_ensembler",
        "logistic_ensembler",
        "neural_network_ensembler",
        "random_forest_ensembler",
        "xgboost_ensembler",
        *[m + "_ensembler" for m in EXACTBOOST],
    ]
    BENCHMARK_MODEL = None

    if METRIC == "auc":
        BENCHMARK_MODEL = "rankboost_ensembler"
    elif METRIC == "ks":
        BENCHMARK_MODEL = "dmks_ensembler"
    elif METRIC == "pak":
        BENCHMARK_MODEL = "top_push_ensembler"

    if BENCHMARK_MODEL:
        MODELS.append(BENCHMARK_MODEL)

DATASETS = args.datasets
TEST_SET = args.test_set

N_SPLITS = args.n_seeds if args.validation else args.n_folds

df = pd.DataFrame()
for dataset in DATASETS:
    for model in MODELS:
        path = f"eval/boxplots-{CV_OR_VAL}/{dataset}/{model}-{CV_OR_VAL}-{TEST_SET}-{N_SPLITS}-{METRIC}.csv"

        if os.path.exists(path):
            data = np.loadtxt(path)
            # By default, we evaluate a metric as gain; transform into loss
            data = 1 - data
            df = df.append([[dataset, model, data.mean(), data.std()]])

df.columns = ["dataset", "model", "mean", "std_dev"]
df = df.reset_index(drop=True)

df = df.round(2)

# Get indices of models with best metric in each dataset evaluation
idx = df.groupby("dataset")["mean"].transform(min) == df["mean"]

# Summarize mean and standard deviation in a single column
df["result"] = df["mean"].astype(str).str.ljust(4, "0").str.cat(
    df["std_dev"].astype(str).str.ljust(4, "0"),
    sep=" \pm "
)
df["result"] = "$" + df["result"] + "$"

# Repeat previous process, but put minimum values in bold
df.loc[idx, "result"] = df["mean"].astype(str).str.ljust(4, "0").str.cat(
    df["std_dev"].astype(str).str.ljust(4, "0"),
    sep="\\boldsymbol{\pm}"
)

df.loc[idx, "result"] = "$\mathbf{" + df["result"] + "}$"

df["dataset"] = df["dataset"].str.replace("_", "\_")
df = df.drop(["mean", "std_dev"], axis=1)

if MODELS_TYPE == "estimators":
    df["model"] = df["model"].replace({
        "adaboost": "AdaBoost",
        "dmks": "DMKS",
        "knn": "kNN",
        "logistic": "Logistic",
        "neural_network": "Neural Net",
        "random_forest": "Rand. For.",
        "rankboost": "RankBoost",
        "top_push": "TopPush",
        "xgboost": "XGBoost",
    })

    # If there is no ambiguity, use ExactBoost as model name
    if len(EXACTBOOST) == 1:
        df["model"] = df["model"].replace(*EXACTBOOST, "ExactBoost")

    # Otherwise, specify loss to distinguish models
    else:
        df["model"] = df["model"].replace({
            "exactboost_auc": "ExactBoost AUC",
            "exactboost_ks":  "ExactBoost KS",
            "exactboost_pak": "ExactBoost P@k",
        })

elif MODELS_TYPE == "ensemblers":
    df["model"] = df["model"].replace({
        "adaboost_ensembler": "AdaBoost",
        "dmks_ensembler": "DMKS",
        "knn_ensembler": "kNN",
        "neural_network_ensembler": "Neural Net",
        "logistic_ensembler": "Logistic",
        "random_forest_ensembler": "Rand. For.",
        "rankboost_ensembler": "RankBoost",
        "top_push_ensembler": "TopPush",
        "xgboost_ensembler": "XGBoost",
    })

    # If there is no ambiguity, use ExactBoost as model name
    if len(EXACTBOOST) == 1:
        df["model"] = df["model"].replace(f"{EXACTBOOST[0]}_ensembler", "ExactBoost")

    # Otherwise, specify loss to distinguish models
    else:
        df["model"] = df["model"].replace({
            "exactboost_auc_ensembler": "ExactBoost AUC",
            "exactboost_ks_ensembler":  "ExactBoost KS",
            "exactboost_pak_ensembler": "ExactBoost P@k",
        })

df = df.rename({"dataset": "Dataset"}, axis=1)

# Transform each model in its own column
df = df.pivot(index="Dataset", columns="model", values="result")
df = df.reset_index().rename_axis(None, axis=1)

# Order columns

exactboost_models = [col for col in df.columns if col.startswith("ExactBoost")]
cols = ["Dataset", *sorted(exactboost_models)]
remaining = [col for col in df.columns if col not in cols]
cols.extend(sorted(remaining, key=str.lower))
if BENCHMARK_MODEL:
    if BENCHMARK_MODEL.startswith("rankboost"):
        cols = sorted(cols, key="RankBoost".__eq__)
    elif BENCHMARK_MODEL.startswith("dmks"):
        cols = sorted(cols, key="DMKS".__eq__)
    elif BENCHMARK_MODEL.startswith("top_push"):
        cols = sorted(cols, key="TopPush".__eq__)
df = df[cols]

# Export as table
if FILENAME is None:
    FILENAME = f"table-{MODELS_TYPE}-{METRIC}.tex"

path = get_folder("eval/tables")

# Fill missing results with NA
df = df.fillna("NA")

df.to_latex(
    f"{path}/{FILENAME}",
    index=False,
    escape=False,
    column_format="c" * df.shape[1]
)

