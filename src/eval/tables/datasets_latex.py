import pandas as pd
from tqdm import tqdm

from src.utils.model_utils import get_data
from src.utils.general_utils import get_folder

# Select datasets
DATASETS = [
    "a1a",
    "australian",
    "banknote",
    "breast-cancer",
    "cod-rna",
    "colon-cancer",
    "covtype",
    "cskaggle",
    "diabetes",
    "fourclass",
    "german",
    "gisette",
    "gmsc",
    "heart",
    "housing",
    "ijcnn1",
    "ionosphere",
    "liver-disorders",
    "madelon",
    "mammography",
    "mq2008",
    "oil-spill",
    "phishing",
    "phoneme",
    "skin-nonskin",
    "sonar",
    "splice",
    "svmguide1",
    "svmguide3",
    "taiwan",
    "w1a",
]

# Create dataframe with datasets characteristics
df = pd.DataFrame()
for dataset in tqdm(DATASETS):
    data = get_data(f"{dataset}_full_train+test")
    df = df.append([
        [dataset, data.shape[0], data.shape[1] - 1, data["target"].mean()]
    ])

df = df.reset_index(drop=True)

df.columns = [
    "Dataset",
    "Observations",
    "Features",
    "Positives"
]

# Set positives as percentage
df["Positives"] = (df["Positives"] * 100).round(1).astype(str) + "%"

# Export table with all datasets for appendix

path = get_folder("eval/tables")

df.to_latex(
    f"{path}/datasets-full.tex",
    index=False,
    column_format="c" * df.shape[1]
)

# Export table with datasets discussed in the main text
MAIN_DATASETS = [
    "a1a",
    "german",
    "gmsc",
    "gisette",
    "heart",
    "ionosphere",
    "liver-disorders",
    "oil-spill",
    "splice",
    "svmguide1",
]

df = df[df["Dataset"].isin(MAIN_DATASETS)]

df.to_latex(
    f"{path}/datasets-main.tex",
    index=False,
    column_format="c" * df.shape[1]
)

