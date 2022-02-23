from argparse import ArgumentParser

import numpy as np
import pandas as pd

from src.models.dmks.model import pestimate
from src.utils.eval_utils import eval_metrics
from src.utils.general_utils import get_folder, save_pandas_to_hdf, tic, toc
from src.utils.model_utils import get_data

with open("data/train_datasets.txt") as file:
    train_datasets = [line.strip() for line in file]
with open("data/test_datasets.txt") as file:
    test_datasets = [line.strip() for line in file]

parser = ArgumentParser()
parser.add_argument(
    "--train_data",
    "-tr",
    help="Which train data to use",
    default="full_train",
    choices=train_datasets,
)
parser.add_argument(
    "--test_data",
    "-te",
    help="Which test data to use",
    default="full_test",
    choices=test_datasets,
)
parser.add_argument(
    "--no-test", action="store_true", help="Use the train data as the test data",
)
parser.add_argument(
    "--n_features", "-N", help="Number of best features to use", default="all",
)
args = parser.parse_args()

if args.no_test:
    args.test_data = args.train_data

# Set variables
TRAIN_DATA = args.train_data
TEST_DATA = args.test_data
N_FEATURES = args.n_features
model_name = "dmks"

print(f"\nLoad {TEST_DATA} data.")
tic()
data = get_data(TEST_DATA, N_FEATURES)
toc()

print(f"\nLoad {model_name} model.")
tic()
model = np.load(f"models/{model_name}/trained_models/model-{TRAIN_DATA}.model")
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
idx = X.index
X = X.to_numpy()
toc()

print(f"\nMake predictions for {TEST_DATA} data.")
tic()
y_pred = pestimate(model["arr_2"], model["arr_1"], X @ model["arr_0"])
preds_df = pd.DataFrame(columns=["preds", "target"], index=idx)
preds_df["preds"] = y_pred
preds_df["target"] = y
toc()

print("\nGet evaluation metrics.")
tic()
print(eval_metrics(preds_df, print_conf_matrix=True))
toc()

print("\nSave predictions.")
tic()
output_folder = get_folder(f"models/{model_name}/preds")
save_pandas_to_hdf(preds_df, f"{output_folder}/model-{TRAIN_DATA}-{TEST_DATA}.h5")
toc()
