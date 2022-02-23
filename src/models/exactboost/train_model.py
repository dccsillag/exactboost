import argparse
import os
import pickle

import numpy as np

from src.models.exactboost.model import ExactBoost, AUC, KS, PaK
from src.utils.general_utils import get_folder, tic, toc
from src.utils.model_utils import get_data

np.random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--metric",
    "-m",
    required=True,
    choices=["auc", "ks", "pak"],
    help="Which metric to use",
)
parser.add_argument(
    "--margin",
    "-ma",
    type=float,
    default=0.05,
    help="Margin to use for training",
)
parser.add_argument(
    "--k",
    "-k",
    type=int,
    help="Value of k to use for pak",
)
parser.add_argument(
    "--beta",
    "-be",
    type=float,
    default=0.2,
    help="Value of beta to use for pak instead of k",
)
parser.add_argument(
    "--train_data",
    "-tr",
    default="heart_full_train",
    help="Which train data to use",
)
parser.add_argument(
    "--n_rounds", "-r", default=50, type=int, help="Number of rounds",
)
parser.add_argument(
    "--observation_subsampling",
    "-os",
    default=0.2,
    type=float,
    help="Proportion of data used in model maximization",
)
parser.add_argument(
    "--feature_subsampling",
    "-fs",
    default=0.2,
    type=float,
    help="Proportion of features to consider in each round",
)
parser.add_argument(
    "--acceptance_rate", "-ar", default=0.0, type=float, help="Acceptance rate"
)
parser.add_argument(
    "--initial_model",
    "-im",
    default="random",
    choices=os.listdir("models/") + ["constant", "random"],
    help="Which model use as initial guess",
)
parser.add_argument(
    "--n_jobs",
    "-j",
    default=-1,
    type=int,
    help="Maximum number of concurrently running jobs",
)
parser.add_argument(
    "--n_estimators",
    "-ne",
    type=int,
    default=250,
    help="Number of estimators in the forest",
)

args = parser.parse_args()

METRIC = args.metric
MARGIN = args.margin
K = args.k
BETA = args.beta
N_ESTIMATORS = args.n_estimators
TRAIN_DATA = args.train_data
N_ROUNDS = args.n_rounds
OBSERVATION_SUBSAMPLING = args.observation_subsampling
FEATURE_SUBSAMPLING = args.feature_subsampling
ACCEPTANCE_RATE = args.acceptance_rate
N_JOBS = args.n_jobs
INITIAL_MODEL = args.initial_model
MODEL_NAME = "exactboost"

print(f"\nLoad {TRAIN_DATA} data.")
tic()
data = get_data(TRAIN_DATA)
data = data.fillna(0) + 0
toc()

print("\nSplit data into features and target.")
tic()
X = data.drop(["target"], axis=1, errors="ignore")
y = np.ravel(data[["target"]])
toc()

print(f"\nFit initial model {INITIAL_MODEL} to train data.")
if K is None:
    k = int(BETA * X.shape[0])
else:
    k = K
if METRIC == "pak":
    print(f"Using k = {k}.")
tic()
if INITIAL_MODEL == "constant":
    initial_score = np.zeros(X.shape[0])
elif INITIAL_MODEL == "random":
    initial_score = lambda: np.random.uniform(size=(X.shape[0],))
else:
    raise ValueError("bad initial model: " + INITIAL_MODEL)
toc()

print(f"\nFit {MODEL_NAME} model to train data.")
tic()
if METRIC == "ks":
    metric = KS()
elif METRIC == "auc":
    metric = AUC()
elif METRIC == "pak":
    metric = PaK(k)
else:
    raise ValueError("bad metric")
model = ExactBoost(
    metric=metric,
    margin=MARGIN,
    n_estimators=N_ESTIMATORS,
    n_rounds=N_ROUNDS,
    observation_subsampling=OBSERVATION_SUBSAMPLING,
    feature_subsampling=FEATURE_SUBSAMPLING,
)

X = X.to_numpy()
model.fit(X, y)
toc()

print("\nSave model.")
tic()
output_folder = get_folder(f"models/{MODEL_NAME}/trained_models")
with open(f"{output_folder}/model-{METRIC}-{INITIAL_MODEL}-{TRAIN_DATA}.pickle", "wb") as file:
    pickle.dump(model, file)
toc()
