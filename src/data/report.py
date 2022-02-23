import pandas as pd
from os import listdir
from os.path import isfile
from tqdm import tqdm

from src.utils.model_utils import get_data

# Read all stored datasets
datasets_folder = "data/processed"
DATASETS = [
    f.split(".")[0]
    for f in listdir(datasets_folder)
    if isfile(f"{datasets_folder}/{f}")
]
DATASETS = sorted(DATASETS)

# Create and print report
report = pd.DataFrame()
for dataset in tqdm(DATASETS):
    df = get_data(f"{dataset}_full_train+test")
    binary = len(df.columns[df.isin([0, 1]).all()]) - 1
    report = report.append([
        [dataset, df.shape[0], df.shape[1] - 1, binary, df["target"].mean()]
    ])

report.columns = [
    "dataset",
    "observations",
    "total features",
    "binary features",
    "proportion of ones"
]

print(report)

# Save report
report.to_csv("data/datasets_report.csv", index=False)
