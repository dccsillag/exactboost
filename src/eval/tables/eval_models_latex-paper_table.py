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
        "german",
        "gisette",
        "gmsc",
        "heart",
        "ionosphere",
        "liver-disorders",
        "oil-spill",
        "splice",
        "svmguide1",
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
parser.add_argument(
    "--only_exact",
    "-oe",
    action='store_true',
    help="Only add exact models"
)
args = parser.parse_args()

CV_OR_VAL = "val" if args.validation else "cv"
FILENAME = args.filename
DATASETS = args.datasets
TEST_SET = args.test_set
MODELS_TYPE = "ensemblers" if args.ensemblers else "estimators"
METRICS = ["ks", "pak", "auc"]
ONLY_EXACT = args.only_exact

df = pd.DataFrame()

for metric in METRICS:
    EXACTBOOST = f"exactboost_{metric}"

    if MODELS_TYPE == "estimators":
        if ONLY_EXACT:
            MODELS = [EXACTBOOST]
        else:
            MODELS = [
                "adaboost",
                "logistic",
                "neural_network",
                "random_forest",
                "xgboost",
                EXACTBOOST,
            ]
        BENCHMARK_MODEL = None

        if metric == "ks":
            BENCHMARK_MODEL = "dmks"
        elif metric == "pak":
            BENCHMARK_MODEL = "top_push"
        elif metric == "auc":
            BENCHMARK_MODEL = "rankboost"

        if BENCHMARK_MODEL:
            MODELS.append(BENCHMARK_MODEL)
            if ONLY_EXACT and metric == "auc":
                MODELS.append("plugin_logistic")
            elif ONLY_EXACT and metric == "pak":
                MODELS.append("svmperf_pak")

    elif MODELS_TYPE == "ensemblers":
        if ONLY_EXACT:
            MODELS = [f"{EXACTBOOST}_ensembler"]
        else:
            MODELS = [
                "adaboost_ensembler",
                "logistic_ensembler",
                "neural_network_ensembler",
                "random_forest_ensembler",
                "xgboost_ensembler",
                f"{EXACTBOOST}_ensembler",
            ]
        BENCHMARK_MODEL = None

        if metric == "ks":
            BENCHMARK_MODEL = "dmks_ensembler"
        elif metric == "pak":
            BENCHMARK_MODEL = "top_push_ensembler"
        elif metric == "auc":
            BENCHMARK_MODEL = "rankboost_ensembler"

        if BENCHMARK_MODEL:
            MODELS.append(BENCHMARK_MODEL)
            if ONLY_EXACT and metric == "auc":
                MODELS.append("plugin_logistic")
            elif ONLY_EXACT and metric == "pak":
                MODELS.append("svmperf_pak_ensembler")

    N_SPLITS = args.n_seeds if args.validation else args.n_folds

    for dataset in DATASETS:
        for model in MODELS:
            path = f"eval/boxplots-{CV_OR_VAL}/{dataset}/{model}-{CV_OR_VAL}-{TEST_SET}-{N_SPLITS}-{metric}.csv"

            if model.endswith("_ensembler"):
                model = model[:-len("_ensembler")]

            if model.startswith("exactboost_"):
                model = "exactboost"
            if not ONLY_EXACT and model in ["dmks", "top_push", "rankboost"]:
                model = "benchmark"

            if os.path.exists(path):
                data = np.loadtxt(path)
                # By default, we evaluate a metric as gain; transform into loss
                data = 1 - data
                df = df.append([[metric, dataset, model, data.mean(), data.std()]])

df.columns = ["metric", "dataset", "model", "mean", "std_dev"]
df = df.reset_index(drop=True)

df = df.round(2)
df["std_dev"] = df["std_dev"].round(1)

# Get indices of models with best metric in each dataset evaluation
idx = df.groupby(["metric", "dataset"])["mean"].transform(min) == df["mean"]

# Summarize mean and standard deviation in a single column
df["result"] = df["mean"].astype(str).str.ljust(4, "0").str.cat(
    df["std_dev"].astype(str).str.ljust(3, "0"),
    sep=" \pm "
)
df["result"] = "$" + df["result"] + "$"

# Repeat previous process, but put minimum values in bold
df.loc[idx, "result"] = df["mean"].astype(str).str.ljust(4, "0").str.cat(
    df["std_dev"].astype(str).str.ljust(3, "0"),
    sep="\\boldsymbol{\pm}"
)

df.loc[idx, "result"] = "$\mathbf{" + df["result"] + "}$"

df["dataset"] = df["dataset"].str.replace("_", "\_")
df = df.drop(["mean", "std_dev"], axis=1)

df["model"] = df["model"].replace({
    "adaboost": "AdaBoost",
    "benchmark": "Exact Bench.",
    "dmks": "DMKS",
    "logistic": "Logistic",
    "neural_network": "Neural Net",
    "random_forest": "Rand. For.",
    "rankboost": "RankBoost",
    "top_push": "TopPush",
    "xgboost": "XGBoost",
    "exactboost": "ExactBoost",
    "plugin_logistic": "Plugin Logistic",
    "svmperf_pak": "SVMPerf",
})

df["dataset"] = df["dataset"].replace({
    "ionosphere": "iono",
    "liver-disorders": "liver",
    "mammography": "mamg.",
    "svmguide1": "svmg1",
})

df["metric"] = df["metric"].replace({
    "auc": "AUC",
    "ks": "KS",
    "pak": "P@k",
})

df = df.rename({"dataset": "Dataset", "metric": "Metric"}, axis=1)

# Transform each model in its own column
if ONLY_EXACT:
    df = df.pivot_table(index="Dataset", columns=["Metric", "model"], values="result", aggfunc="first")
    df = df[[("AUC", "ExactBoost"), ("AUC", "RankBoost"), ("AUC", "Plugin Logistic"),
             ("KS", "ExactBoost"), ("KS", "DMKS"),
             ("P@k", "ExactBoost"), ("P@k", "TopPush"), ("P@k", "SVMPerf")]]
    df = df.reset_index().rename_axis([None, None], axis=1)
    n_columns = df.shape[1]

    # Set dataset as index
    df.index = df["Dataset"]
    df = df.drop(["Dataset"], axis=1)
else:
    df = df.pivot_table(index=["Metric", "Dataset"], columns="model", values="result", aggfunc="first")
    df = df.reset_index().rename_axis(None, axis=1)
    n_columns = df.shape[1]

    # Set metric (loss) and dataset as indices
    df = df.rename({"Metric": "Loss"}, axis=1)
    df.index = pd.MultiIndex.from_frame(df[["Loss", "Dataset"]])
    df = df.drop(["Loss", "Dataset"], axis=1)

# Order columns
cols = ["ExactBoost"]
remaining = [col for col in df.columns if col not in cols]
if ONLY_EXACT:
    cols.extend(sorted(remaining, key=lambda x: x[1].lower()))
else:
    cols.extend(sorted(remaining, key=str.lower))
    cols = sorted(cols, key="Exact Bench.".__eq__)
    df = df[cols]

# Fill missing results with NA
df = df.fillna("NA")

# Export as table
if FILENAME is None:
    FILENAME = f"table-{MODELS_TYPE}-all_metrics-main.tex"

path = get_folder("eval/tables")

df.to_latex(
    f"{path}/{FILENAME}",
    multirow=True,
    index=True,
    sparsify=True,
    escape=False,
    column_format="c" * n_columns
)

print(df)
