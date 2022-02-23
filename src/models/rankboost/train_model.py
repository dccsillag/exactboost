import argparse
import pickle

import numpy as np

from src.models.rankboost.model import RankBoost
from src.utils.general_utils import get_folder, tic, toc
from src.utils.model_utils import get_X_y

with open("data/train_datasets.txt") as file:
    train_datasets = [line.strip() for line in file]
with open("data/test_datasets.txt") as file:
    test_datasets = [line.strip() for line in file]

np.random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--train_data",
    "-tr",
    default="heart_full_train",
    choices=train_datasets,
    help="Which train data to use",
)
parser.add_argument(
    "--n_rounds", "-r", default=10, type=int, help="Number of rounds",
)
args = parser.parse_args()

MODEL_NAME = "rankboost"
TRAIN_DATA = args.train_data
N_ROUNDS = args.n_rounds

print(f"\nLoad {TRAIN_DATA} data.")
tic()
X, y = get_X_y(TRAIN_DATA)
toc()

print(f"\nFit {MODEL_NAME} model to train data.")
tic()
model = RankBoost(n_rounds=N_ROUNDS)

model.fit(X, y)

print("\nSave model.")
tic()
output_folder = get_folder(f"models/{MODEL_NAME}/trained_models")
with open(f"{output_folder}/model-{TRAIN_DATA}.pickle", 'wb') as file:
    pickle.dump(model, file)
toc()
