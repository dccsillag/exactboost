from math import ceil
import argparse
import subprocess as sp

import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from src.models.exactboost.model import ExactBoost, AUC, KS, PaK
from src.utils.eval_utils import ks_np, precision_at_k
from src.utils.general_utils import get_folder
from src.utils.model_utils import get_X_y

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", "-d")
parser.add_argument("--metric", "-m", default="ks")
parser.add_argument("--observation_subsampling", "-os", type=float, default=0.2)
parser.add_argument("--feature_subsampling", "-fs", type=float, default=0.2)
parser.add_argument("--n_rounds", "-r", type=int, default=50)
parser.add_argument("--n_estimators", "-e", type=int, default=250)
args = parser.parse_args()

np.random.seed(0)

DATASET = args.dataset
METRIC = args.metric
SPLIT_SEED = 0

X_train, y_train = get_X_y(f"{DATASET}_full_train", seed=SPLIT_SEED)
X_test, y_test = get_X_y(f"{DATASET}_full_test", seed=SPLIT_SEED)

N_ESTIMATORS = args.n_estimators
N_ROUNDS = args.n_rounds
OBS_SUBSAMPLING = args.observation_subsampling
FEAT_SUBSAMPLING = args.feature_subsampling
BETA = y_train.mean()
K_TRAIN = int(np.ceil(BETA * len(y_train)))
K_TEST = int(np.ceil(BETA * len(y_test)))

N_SAMPLES = 192
N_JOBS = 16

thetas = np.linspace(0, 0.7, N_SAMPLES)

def run_exactboostmargin(theta):
    # Train model
    np.random.seed(0)
    metric = {
        "ks": KS(),
        "auc": AUC(),
        "pak": PaK(ceil(X_train.shape[0] * BETA)),
    }[METRIC]
    model = ExactBoost(
        metric=metric,
        n_estimators=N_ESTIMATORS,
        n_rounds=N_ROUNDS,
        margin=theta,
        observation_subsampling=OBS_SUBSAMPLING,
        feature_subsampling=FEAT_SUBSAMPLING,
    )

    model.fit(X_train, y_train)

    # Predict
    train_predicitons = model.predict(X_train)
    test_predicitons = model.predict(X_test)

    # Eval model
    if METRIC == "auc":
        return roc_auc_score(y_train, train_predicitons), roc_auc_score(y_test, test_predicitons)
    elif METRIC == "ks":
        return ks_np(y_train, train_predicitons), ks_np(y_test, test_predicitons)
    elif METRIC == "pak":
        return precision_at_k(y_train, train_predicitons, k=K_TRAIN)[0] / K_TRAIN, precision_at_k(y_test, test_predicitons, k=K_TEST)[0] / K_TEST


train_metric, test_metric = zip(
    *Parallel(n_jobs=N_JOBS)(
        delayed(run_exactboostmargin)(theta) for theta in tqdm(thetas)
    )
)

# By default, we evaluate a metric as gain; transform into loss
train_loss = 1 - np.array(train_metric)
test_loss = 1 - np.array(test_metric)

path = get_folder("eval/margin_plots")

np.save(
    f"{path}/{METRIC}_margin-{DATASET}-train-os_{OBS_SUBSAMPLING}-fs_{FEAT_SUBSAMPLING}-"
    f"r_{N_ROUNDS}-estimators_{N_ESTIMATORS}-split_{SPLIT_SEED}.npy",
    {tuple(thetas): train_loss},
)
np.save(
    f"{path}/{METRIC}_margin-{DATASET}-test-os_{OBS_SUBSAMPLING}-fs_{FEAT_SUBSAMPLING}-"
    f"r_{N_ROUNDS}-estimators_{N_ESTIMATORS}-split_{SPLIT_SEED}.npy",
    {tuple(thetas): test_loss},
)

sp.call([
    "python3", "src/eval/margin_plots/plot.py",
    "-d", f"{DATASET}",
    "-m", f"{METRIC}",
    "-os", f"{OBS_SUBSAMPLING}",
    "-fs", f"{FEAT_SUBSAMPLING}",
    "-r", f"{N_ROUNDS}",
    "-e", f"{N_ESTIMATORS}",
])
