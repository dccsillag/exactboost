import socket
from argparse import ArgumentParser

import numpy as np
import pandas as pd

from src.utils.general_utils import get_folder
from src.utils.model_utils import get_X_y


parser = ArgumentParser()
parser.add_argument("-m", "--models",
                    nargs="+",
                    default=["dmks:exactboost_ks", "rankboost:exactboost_auc", "toppush:exactboost_pak", "xgboost:exactboost_ks"],
                    help="Models to add to the table")
parser.add_argument("-d", "--datasets",
                    nargs="+",
                    help="Datasets to add to the table")
parser.add_argument("-n", "--n_samples",
                    type=int,
                    default=5,
                    help="Use results that were run with the given number of samples")
parser.add_argument("-j", "--n_jobs",
                    type=int,
                    default=-1,
                    help="Use results that were run using the given number of threads")
parser.add_argument("-H", "--hostname",
                    default=socket.gethostname(),
                    help="Use results that were run on the given machine")
parser.add_argument("-f", "--first_only",
                    action="store_true",
                    help="Only show the first model in the header")
parser.add_argument("-D", "--diffs",
                    action="store_true",
                    help="Use time differences instead of proportions")
args = parser.parse_args()


MODEL_HUMAN_NAMES = {
    "exactboost_auc": "E.B.",
    "exactboost_ks": "E.B.",
    "exactboost_pak": "E.B.",
    "adaboost": "AdaBoost",
    "dmks": "DMKS",
    "knn": "kNN",
    "logistic": "Logistic",
    "neural_network": "Neural Net",
    "random_forest": "Rand. For.",
    "rankboost": "RankBoost",
    "toppush": "TopPush",
    "xgboost": "XGBoost",
}


def get_name(model_pair):
    model0, model1 = model_pair.split(":")
    if args.first_only:
        return MODEL_HUMAN_NAMES[model0]
    else:
        if args.diffs:
            return MODEL_HUMAN_NAMES[model0] + "-" + MODEL_HUMAN_NAMES[model1]
        else:
            return MODEL_HUMAN_NAMES[model0] + "/" + MODEL_HUMAN_NAMES[model1]


df = pd.DataFrame(columns=["Dataset", "Observations", "Features", "Positives"]
                  + [get_name(x) for x in args.models])
df.reset_index(drop=True)

for dataset in args.datasets:
    X, y = get_X_y(f"{dataset}_full_train+test")

    new_row = {
        "Dataset": dataset,
        "Observations": X.shape[0],
        "Features": X.shape[1],
        "Positives": y.mean(),
    }

    for model_pair in args.models:
        model0, model1 = model_pair.split(":")
        try:
            measurements0 = pd.read_csv("eval/running_time_benchmarks/"
                                        f"{model0}-{dataset}-{args.n_jobs}-{args.n_samples}-{args.hostname}.csv",
                                        header=None)
            measurements1 = pd.read_csv("eval/running_time_benchmarks/"
                                        f"{model1}-{dataset}-{args.n_jobs}-{args.n_samples}-{args.hostname}.csv",
                                        header=None)
            if args.diffs:
                new_row[get_name(model_pair)] = measurements0[0].mean() - measurements1[0].mean()
            else:
                new_row[get_name(model_pair)] = measurements0[0].mean() / measurements1[0].mean()
        except FileNotFoundError:
            new_row[get_name(model_pair)] = np.NaN

    df = df.append(new_row, ignore_index=True)

df["Positives"] = (df["Positives"] * 100).round(2).astype(str) + "%"
for model_pair in args.models:
    colname = get_name(model_pair)
    if args.diffs:
        df[colname] = df[colname].round(1).map("{:+,.1f}s".format)
    else:
        df[colname] = df[colname].round(1).map("{:.1f}x".format)

df = df.sort_values("Dataset")

path = get_folder("eval/tables")
df.to_latex(
    f"{path}/datasets-with-benchmarks.tex",
    index=False,
    column_format="cccc|" + "r"*len(args.models)
)
