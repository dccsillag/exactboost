import os
import time
import socket
from argparse import ArgumentParser
import multiprocessing
from math import ceil

from tqdm import trange
import numpy as np
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import tensorflow as tf
from src.models.dmks.model import imo
from julia import Julia

from src.utils.model_utils import get_X_y
from src.models.exactboost.model import ExactBoost, AUC, KS, PaK
from src.models.rankboost.model import RankBoost

# Use a custom system image, with PyCall (1) and ClassificationOnTop (2) compiled in.
# (1) is a workaround needed due to python being installed through Conda.
# (2) greatly reduces initialization time.
jl = Julia(
    sysimage="setup/julia-1.5.3/custom_sysimage.so", compiled_modules=False
)
from julia import ClassificationOnTop



parser = ArgumentParser()
parser.add_argument("-m", "--model",
                    required=True,
                    choices=[
                        "exactboost_auc",
                        "exactboost_ks",
                        "exactboost_pak",
                        "adaboost",
                        "knn",
                        "logistic",
                        "xgboost",
                        "random_forest",
                        "neural_network",
                        "rankboost",
                        "dmks",
                        "top_push",
                    ],
                    help="Model to benchmark")
parser.add_argument("-d", "--dataset",
                    help="Dataset to benchmark")
parser.add_argument("-n", "--n_samples",
                    type=int,
                    default=5,
                    help="How many times to sample the running time")
parser.add_argument("-j", "--n_jobs",
                    type=int,
                    default=-1,
                    help="How many jobs to use")
args = parser.parse_args()


X, y = get_X_y(f"{args.dataset}_full_train")
beta = get_X_y(f"{args.dataset}_full_train+test")[1].mean()
k = ceil(beta*len(y))

time_measurements = np.empty(args.n_samples)

for i in trange(args.n_samples):
    start_time = time.time()

    if args.model == "exactboost_ks":
        metric = KS()
        ExactBoost(metric).fit(X, y, interaction=False)
    elif args.model == "exactboost_auc":
        metric = AUC()
        model = ExactBoost(metric).fit(X, y, interaction=False)
    elif args.model == "exactboost_pak":
        metric = PaK(k)
        ExactBoost(metric).fit(X, y, interaction=False)
    elif args.model == "adaboost":
        AdaBoostClassifier(random_state=0).fit(X, y)
    elif args.model == "knn":
        KNeighborsClassifier(n_jobs=args.n_jobs).fit(X, y)
    elif args.model == "logistic":
        LogisticRegression(random_state=0, n_jobs=args.n_jobs).fit(X, y)
    elif args.model == "xgboost":
        XGBClassifier(seed=0, n_jobs=args.n_jobs).fit(X, y)
    elif args.model == "random_forest":
        RandomForestClassifier(random_state=0, n_jobs=args.n_jobs).fit(X, y)
    elif args.model == "neural_network":
        n_cpus = multiprocessing.cpu_count()
        assert args.n_jobs == n_cpus, f"Neural networks can only be trained with `-j {n_cpus}`"
        tf.random.set_seed(0)
        model = tf.keras.models.Sequential(
            [
                tf.keras.layers.Dense(26, activation="relu"),
                tf.keras.layers.Dense(12, activation="relu"),
                tf.keras.layers.Dense(12, activation="relu"),
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ]
        )
        model.compile(optimizer="adam", loss="binary_crossentropy")
        model.fit(X, y, epochs=30, batch_size=4)
    elif args.model == "rankboost":
        RankBoost(n_rounds=100).fit(X, y)
    elif args.model == "dmks":
        X = X + np.finfo(float).eps
        init_model = LogisticRegression(random_state=0, n_jobs=args.n_jobs)
        init_model.fit(X, y)
        beta_zero = init_model.coef_[0] / np.linalg.norm(init_model.coef_[0])
        model = imo(X, y, weight=np.ones(X.shape[1]), beta_start=beta_zero, lamb=0)[0]
    elif args.model == "top_push":
        alg = ClassificationOnTop.TopPush()
        w_ini = np.zeros(X.shape[1])
        model, _, _ = ClassificationOnTop.solver(alg, (X, y), w=w_ini)
    else:
        raise RuntimeError(f"Bad model: {args.model}")

    end_time = time.time()

    time_measurements[i] = end_time - start_time

hostname = socket.gethostname()
if not os.path.exists("eval/running_time_benchmarks"):
    os.makedirs("eval/running_time_benchmarks")
np.savetxt(f"eval/running_time_benchmarks/{args.model}-{args.dataset}-{args.n_jobs}-{args.n_samples}-{hostname}.csv",
           time_measurements)
