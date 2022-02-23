import os
from argparse import ArgumentParser

import numpy as np

from src.models.dmks.model import imo
from src.utils.general_utils import get_folder, tic, toc
from src.utils.model_utils import get_data

with open("data/train_datasets.txt") as file:
    train_datasets = [line.strip() for line in file]

parser = ArgumentParser()
parser.add_argument(
    "--train_data",
    "-tr",
    help="Which train data to use",
    default="full_train",
    choices=train_datasets,
)
parser.add_argument(
    "--n_features", "-N", help="Number of feaures to use", default="all",
)

args = parser.parse_args()

TRAIN_DATA = args.train_data
N_FEATURES = args.n_features
model_name = "dmks"

print(f"\nLoad {TRAIN_DATA} data.")
tic()
data = get_data(TRAIN_DATA, N_FEATURES)
toc()

print("\nSplit data into features and target.")
tic()
X = data.drop(["target"], axis=1)
y = np.ravel(data[["target"]])
toc()

print("\nProcess data to deal with missing values and division by zero.")
tic()
X = X.fillna(0)
X = X + np.finfo(float).eps
X = X.to_numpy()
toc()

print(f"\nFit {model_name} model to training data.")
tic()
beta_zero = np.linalg.lstsq(X, y, rcond=None)[0]
beta_zero = beta_zero / np.linalg.norm(beta_zero)
model = imo(X, y, weight=np.ones(X.shape[1]), beta_start=beta_zero, lamb=0)[0]
toc()

print("\nSave model.")
tic()
output_folder = get_folder(f"models/{model_name}/trained_models")
np.savez(f"{output_folder}/model-{TRAIN_DATA}.model", model, y, X @ model)
# savez appends .npz extension to files and there is no way to prevent it
# manually rename model
os.rename(
    f"{output_folder}/model-{TRAIN_DATA}.model.npz",
    f"{output_folder}/model-{TRAIN_DATA}.model",
)
print(f"-saved model {output_folder}/model-{TRAIN_DATA}.model")
toc()
