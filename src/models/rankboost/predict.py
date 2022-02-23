from argparse import ArgumentParser
import pickle

import pandas as pd

from src.utils.eval_utils import eval_metrics
from src.utils.general_utils import (
    get_folder,
    save_pandas_to_hdf,
    tic,
    toc,
)
from src.utils.model_utils import get_X_y

with open("data/train_datasets.txt") as file:
    train_datasets = [line.strip() for line in file]
with open("data/test_datasets.txt") as file:
    test_datasets = [line.strip() for line in file]

parser = ArgumentParser()
parser.add_argument(
    "--train_data",
    "-tr",
    default="heart_full_train",
    choices=train_datasets,
    help="Which train data to use",
)
parser.add_argument(
    "--test_data",
    "-te",
    default="heart_full_test",
    choices=test_datasets + train_datasets,
    help="Which test data to use",
)
parser.add_argument(
    "--no-test", action="store_true", help="Use the train data as the test data",
)
args = parser.parse_args()

if args.no_test:
    args.test_data = args.train_data

assert (
    args.train_data.split("_")[0] == args.test_data.split("_")[0]
), "please use the correct test dataset"

# Set variables
TRAIN_DATA = args.train_data
TEST_DATA = args.test_data
MODEL_NAME = "rankboost"


print(f"\nLoad {TEST_DATA} data.")
tic()
X, y = get_X_y(TEST_DATA)
toc()

print(f"\nLoad {MODEL_NAME} model")
tic()
with open(f"models/rankboost/trained_models/model-{TRAIN_DATA}.pickle", 'rb') as file:
    model = pickle.load(file)
toc()

print(f"\nMake predictions for {TEST_DATA} data using RankBoost")
tic()

y_pred = model.predict(X)
preds_df = pd.DataFrame(columns=["preds", "target"])
preds_df["preds"] = y_pred
preds_df["target"] = y
toc()

print("\nGet evaluation metrics for RankBoost model.")
tic()
print(eval_metrics(preds_df, print_conf_matrix=True))
toc()

print("\nSave predictions.")
tic()
output_folder = get_folder(f"models/{MODEL_NAME}/preds")
save_pandas_to_hdf(preds_df, f"{output_folder}/model-{TRAIN_DATA}-{TEST_DATA}.h5")
toc()
