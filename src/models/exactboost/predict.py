import os
from argparse import ArgumentParser
import pickle

import numpy as np
import pandas as pd

from src.utils.eval_utils import eval_metrics
from src.utils.general_utils import tic, toc
from src.utils.model_utils import get_data

parser = ArgumentParser()
parser.add_argument(
    "--metric",
    "-m",
    required=True,
    choices=["auc", "ks", "pak"],
    help="Which metric to use",
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
    "--test_data",
    "-te",
    default="heart_full_test",
    help="Which test data to use",
)
parser.add_argument(
    "--no-test", action="store_true", help="Use the train data as the test data",
)
parser.add_argument(
    "--initial_model",
    "-im",
    default="random",
    choices=os.listdir("models/") + ["constant", "random"],
    help="Which model use as initial guess",
)

args = parser.parse_args()

if args.no_test:
    args.test_data = args.train_data

assert (
    args.train_data.split("_")[0] == args.test_data.split("_")[0]
), "please use the correct test dataset"

# Set variables
METRIC = args.metric
K = args.k
BETA = args.beta
TRAIN_DATA = args.train_data
TEST_DATA = args.test_data
INITIAL_MODEL = args.initial_model
MODEL_NAME = "exactboost"

print(f"\nLoad {TEST_DATA} data.")
tic()
data = get_data(TEST_DATA)
data = data.fillna(0) + 0
toc()

print("\nSplit data into features and target.")
tic()
X = data.drop(["target"], axis=1)
y = np.ravel(data[["target"]])
toc()

print(f"\nFit initial model {INITIAL_MODEL} to test data.")
tic()
if INITIAL_MODEL == "constant":
    initial_score = np.zeros(X.shape[0])
elif INITIAL_MODEL == "random":
    initial_score = np.random.uniform(size=(X.shape[0],))
else:
    raise ValueError("bad initial model: " + INITIAL_MODEL)
toc()

print(f"\nLoad {MODEL_NAME} model")
tic()
with open(f"models/exactboost/trained_models/model-{METRIC}-{INITIAL_MODEL}-{TRAIN_DATA}.pickle", "rb") as file:
    model = pickle.load(file)
toc()

print(f"\nMake predictions for {TEST_DATA} data using ExactBoost")
tic()
index = X.index
X = X.to_numpy()

sb_y_pred = model.predict(X)
sb_preds_df = pd.DataFrame(columns=["preds", "target"], index=index)
sb_preds_df["preds"] = sb_y_pred
sb_preds_df["target"] = y
toc()

print("\nGet evaluation metrics for ExactBoost model.")
tic()
print(eval_metrics(sb_preds_df, k=K, beta=BETA, print_conf_matrix=True))
toc()
