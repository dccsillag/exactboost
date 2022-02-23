import argparse
import numpy as np
import pandas as pd
import pandas.api.types as ptypes
from sklearn.datasets import load_svmlight_file

from src.utils.data_utils import drop_missing
from src.utils.general_utils import get_folder


def process(dataset_name):
    """Given dataset name, load it and process it into a DataFrame ready to be sent
    to one of the models (i.e., only numerical columns).

    Args:
        dataset_name (str): identifier for given dataset.

    Returns:
        pd.DataFrame: processed DataFrame.
    """

    if dataset_name in [
        "a1a",
        "a2a",
        "a3a",
        "a4a",
        "a5a",
        "a6a",
        "a7a",
        "a8a",
        "a9a",
        "avazu",
    ]:
        X, y = load_svmlight_file(f"data/raw/{dataset_name}/{dataset_name}")
        data = pd.DataFrame(X.toarray())
        data["target"] = 0.5 * (pd.DataFrame(y) + 1)

    elif dataset_name == "australian":
        data = pd.read_csv(
            f"data/raw/{dataset_name}/australian.dat", sep=" ", header=None,
        )
        cols = [f"A{n}" for n in range(1,data.shape[1])]
        cols.append("target")
        data.columns = cols

    elif dataset_name == "banknote":
        data = pd.read_table(
            f"data/raw/{dataset_name}/data_banknote_authentication.txt",
            sep=",",
            header=None,
            names=["variance", "skewness", "kurtosis", "entropy", "target"],
        )

    elif dataset_name == "breast-cancer":
        X, y = load_svmlight_file(f"data/raw/{dataset_name}/{dataset_name}")
        data = pd.DataFrame(X.toarray())
        data["target"] = 0.5 * (pd.DataFrame(y) - 2)

    elif dataset_name == "cod-rna":
        X, y = load_svmlight_file(f"data/raw/{dataset_name}/{dataset_name}")
        data = pd.DataFrame(X.toarray())
        data["target"] = 0.5 * (pd.DataFrame(y) + 1)

    elif dataset_name == "colon-cancer":
        X, y = load_svmlight_file(f"data/raw/{dataset_name}/colon-cancer.bz2")
        data = pd.DataFrame(X.toarray())
        data["target"] = 0.5 * (pd.DataFrame(y) + 1)

    elif dataset_name == "covtype":
        X, y = load_svmlight_file(f"data/raw/{dataset_name}/covtype.libsvm.binary.bz2")
        data = pd.DataFrame(X.toarray())
        data["target"] = pd.DataFrame(y) - 1

    elif dataset_name == "cskaggle":
        # Read data and add NA code
        data = pd.read_csv(f"data/raw/{dataset_name}/application_train.csv")
        data["DAYS_EMPLOYED"] = data["DAYS_EMPLOYED"].replace({365243: np.nan})
        # Exclude columns with >40% NAs, categorical with too many categories, and
        # convert non-numerical columns to dummies
        data = drop_missing(data, missing_fraction=0.4)
        data = data.drop(["SK_ID_CURR", "OCCUPATION_TYPE", "ORGANIZATION_TYPE"], axis=1)
        non_numeric_columns = [
            data.columns[i] for i, x in enumerate(data.dtypes) if x == "O"
        ]
        data = pd.get_dummies(data, non_numeric_columns, drop_first=True)
        # Rename target columns
        data = data.rename(columns={"TARGET": "target"})

    elif dataset_name == "diabetes":
        data = pd.read_csv(
            f"data/raw/{dataset_name}/pima-indians-diabetes.data.csv",
            header=None,
            names=[
                "times_pregnant",
                "glucose",
                "diastolic_pressure",
                "thickness",
                "insulin",
                "bmi",
                "pedigree",
                "age",
                "target",
            ],
        )

    elif dataset_name == "epsilon":
       X, y = load_svmlight_file(f"data/raw/{dataset_name}/epsilon_normalized.bz2")
       data = pd.DataFrame(X.toarray())
       data["target"] = 0.5 * (pd.DataFrame(y) + 1)

    elif dataset_name == "duke-breast-cancer":
        X, y = load_svmlight_file(f"data/raw/{dataset_name}/duke.bz2")
        data = pd.DataFrame(X.toarray())
        data["target"] = 0.5 * (1 - pd.DataFrame(y))

    elif dataset_name == "fourclass":
        X, y = load_svmlight_file(f"data/raw/{dataset_name}/fourclass")
        data = pd.DataFrame(X.toarray())
        data["target"] = 0.5 * (pd.DataFrame(y) + 1)

    elif dataset_name == "german":
        # Raw data comes with double header; keep only one
        data = pd.read_csv(
            f"data/raw/{dataset_name}/german_credit.csv", sep=",", header=0,
        )
        # Fix ID column name
        data = data.rename(columns={"Creditability": "target"})

    elif dataset_name == "gisette":
        X, y = load_svmlight_file(f"data/raw/{dataset_name}/gisette_scale.bz2")
        data = pd.DataFrame(X.toarray())
        data["target"] = 0.5 * (pd.DataFrame(y) + 1)

    elif dataset_name == "gmsc":
        data = pd.read_csv(f"data/raw/{dataset_name}/cs-training.csv",)
        # Drop ID column
        data = data.drop("Unnamed: 0", axis=1)
        data = data.rename(columns={"SeriousDlqin2yrs": "target"})

    elif dataset_name == "heart":
        data = pd.read_table(
            f"data/raw/{dataset_name}/processed.cleveland.data",
            sep=",",
            header=None,
            index_col=False,
            names=[
                "age",
                "sex_is_male",
                "chest_pain_type",
                "blood_pressure",
                "cholesterol",
                "blood_sugar_is_above_120",
                "ecg",
                "maximum_hr",
                "has_angina",
                "oldpeak",
                "slope_exercise",
                "number_of_colored_vessels",
                "thal",
                "target",
            ],
        )
        data["number_of_colored_vessels"] = pd.to_numeric(
            data["number_of_colored_vessels"].replace({"?": 0})
        ).astype(int)
        data = pd.get_dummies(data, columns=["chest_pain_type", "ecg", "thal"])
        data["target"] = np.where(data["target"] > 0, 1, 0)

    elif dataset_name == "higgs":
        X, y = load_svmlight_file(f"data/raw/{dataset_name}/HIGGS.bz2")
        data = pd.DataFrame(X.toarray())
        data["target"] = pd.DataFrame(y)

    elif dataset_name == "housing":
        data = pd.read_table(
            f"data/raw/{dataset_name}/housing.data",
            delim_whitespace=True,
            header=None,
            names=[
                "CRIM",
                "ZN",
                "INDUS",
                "CHAS",
                "NOX",
                "RM",
                "AGE",
                "DIS",
                "RAD",
                "TAX",
                "PTRATIO",
                "B",
                "LSTAT",
                "MEDV",
            ],
        )
        data = data.rename(columns={"CHAS": "target"})

    elif dataset_name == "ijcnn1":
        X, y = load_svmlight_file(f"data/raw/{dataset_name}/ijcnn1.bz2")
        data = pd.DataFrame(X.toarray())
        data["target"] = 0.5 * (pd.DataFrame(y) + 1)

    elif dataset_name == "ionosphere":
        data = pd.read_table(
            f"data/raw/{dataset_name}/ionosphere.csv",
            sep=",",
            header=None,
            names=[
                f"pulse_{number}_{type}"
                for number in range(17)
                for type in ["real", "complex"]
            ]
            + ["target"],
        )
        data["target"] = np.where(data["target"] == "g", 1, 0)

    elif dataset_name == "kdd":
        # Load features
        features = pd.read_csv(
            f"data/raw/{dataset_name}/Features.txt", sep="\t", header=None
        )
        # Drop last column as it is empty
        features = features.dropna(axis=1)
        # Load target variable
        y = pd.read_csv(
            f"data/raw/{dataset_name}/Info.txt", sep="\t", header=None, usecols=[0]
        )
        y = y.rename({0: "target"}, axis=1)
        # Map [-1, 1] into [0, 1]
        y = (y + 1) * 0.5
        # Convert target variable to integer
        y = y.astype(int)
        # Concatenate features and target
        data = pd.concat([features, y], axis=1)

    elif dataset_name == "liver-disorders":
        X, y = load_svmlight_file(f"data/raw/{dataset_name}/liver-disorders")
        data = pd.DataFrame(X.toarray())
        data["target"] = pd.DataFrame(y)

    elif dataset_name == "madelon":
        X, y = load_svmlight_file(f"data/raw/{dataset_name}/madelon")
        data = pd.DataFrame(X.toarray())
        data["target"] = 0.5 * (pd.DataFrame(y) + 1)

    elif dataset_name == "mammography":
        data = pd.read_table(
            f"data/raw/{dataset_name}/mammography.csv",
            sep=",",
            header=None,
            names=[
                "area",
                "gray_level",
                "gradient_strength",
                "rms_noise",
                "contrast",
                "moment",
                "target",
            ],
        )
        data["target"] = np.where(data["target"] == "'1'", 1, 0)

    elif dataset_name == "mq2008":
        # Read and process data to disregard comments (#)
        # and built-in column names (:)
        data = pd.read_table(
            f"data/raw/{dataset_name}/min.txt",
            sep="\s+|:",
            header=None,
            comment="#",
            engine="python",
        )
        cols_to_ignore = [2] + [2 * i + 1 for i in range(47)]
        data = data.drop(cols_to_ignore, axis=1)
        # Rename columns, designate target column by "target"
        data.columns = ["target"] + list(range(1, 46 + 1))
        # Transform problem into binary classification by making
        # documents with relevance bigger than 0 equal to 1
        data["target"] = np.where(data["target"] > 0, 1, 0)

    elif dataset_name == "mushrooms":
        X, y = load_svmlight_file(f"data/raw/{dataset_name}/mushrooms")
        data = pd.DataFrame(X.toarray())
        data["target"] = pd.DataFrame(y) - 1

    elif dataset_name == "news20":
        X, y = load_svmlight_file(f"data/raw/{dataset_name}/news20.binary.bz2")
        data = pd.DataFrame(X.toarray())
        data["target"] = 0.5 * (pd.DataFrame(y) + 1)

    elif dataset_name == "oil-spill":
        data = pd.read_table(
            f"data/raw/{dataset_name}/oil-spill.csv",
            sep=",",
            header=None,
            names=[f"feat_{i}" for i in range(49)] + ["target"],
        )

    elif dataset_name == "phishing":
        X, y = load_svmlight_file(f"data/raw/{dataset_name}/phishing")
        data = pd.DataFrame(X.toarray())
        data["target"] = pd.DataFrame(y)

    elif dataset_name == "phoneme":
        data = pd.read_table(
            f"data/raw/{dataset_name}/phoneme.csv",
            sep=",",
            header=None,
            names=[f"amplitude_{i}" for i in range(5)] + ["target"],
        )

    elif dataset_name == "real-sim":
        X, y = load_svmlight_file(f"data/raw/{dataset_name}/real-sim.bz2")
        data = pd.DataFrame(X.toarray())
        data["target"] = 0.5 * (pd.DataFrame(y) + 1)

    elif dataset_name == "skin-nonskin":
        X, y = load_svmlight_file(f"data/raw/{dataset_name}/skin_nonskin")
        data = pd.DataFrame(X.toarray())
        data["target"] = pd.DataFrame(y) - 1

    elif dataset_name == "sonar":
        data = pd.read_table(
            f"data/raw/{dataset_name}/sonar.csv",
            sep=",",
            header=None,
            names=[f"signal_{s}" for s in range(60)] + ["target"],
        )
        data["target"] = np.where(data["target"] == "R", 1, 0)

    elif dataset_name == "splice":
        X, y = load_svmlight_file(f"data/raw/{dataset_name}/splice")
        data = pd.DataFrame(X.toarray())
        data["target"] = 0.5 * (1 - pd.DataFrame(y))

    elif dataset_name == "svmguide1":
        X, y = load_svmlight_file(f"data/raw/{dataset_name}/svmguide1")
        data = pd.DataFrame(X.toarray())
        data["target"] = 1 - pd.DataFrame(y)

    elif dataset_name == "svmguide3":
        X, y = load_svmlight_file(f"data/raw/{dataset_name}/svmguide3")
        data = pd.DataFrame(X.toarray())
        data["target"] = 0.5 * (pd.DataFrame(y) + 1)

    elif dataset_name == "taiwan":
        # Raw data comes with double header; keep only one
        data = pd.read_excel(
            f"data/raw/{dataset_name}/default%20of%20credit%20card%20clients.xls",
            sep=";",
            header=0,
            skiprows=1,
        )
        # Fix target column name
        data = data.rename(columns={"default payment next month": "target"})
        # Process data like in Chen and Fang (2019)
        data["SEX"] = data["SEX"].replace(2, 0)
        data["EDUCATION"] = data["EDUCATION"].replace(2, 1)
        data["EDUCATION"] = data["EDUCATION"].replace([3, 4, 5, 6], 0)
        data["MARRIAGE"] = data["MARRIAGE"].replace([2, 3], 0)

    elif dataset_name == "w1a":
        X, y = load_svmlight_file(f"data/raw/{dataset_name}/w1a")
        data = pd.DataFrame(X.toarray())
        data["target"] = 0.5 * (pd.DataFrame(y) + 1)

    elif dataset_name == "w2a":
        X, y = load_svmlight_file(f"data/raw/{dataset_name}/w2a")
        data = pd.DataFrame(X.toarray())
        data["target"] = 0.5 * (pd.DataFrame(y) + 1)

    elif dataset_name == "w3a":
        X, y = load_svmlight_file(f"data/raw/{dataset_name}/w3a")
        data = pd.DataFrame(X.toarray())
        data["target"] = 0.5 * (pd.DataFrame(y) + 1)

    elif dataset_name == "w4a":
        X, y = load_svmlight_file(f"data/raw/{dataset_name}/w4a")
        data = pd.DataFrame(X.toarray())
        data["target"] = 0.5 * (pd.DataFrame(y) + 1)

    elif dataset_name == "w5a":
        X, y = load_svmlight_file(f"data/raw/{dataset_name}/w5a")
        data = pd.DataFrame(X.toarray())
        data["target"] = 0.5 * (pd.DataFrame(y) + 1)

    elif dataset_name == "w6a":
        X, y = load_svmlight_file(f"data/raw/{dataset_name}/w6a")
        data = pd.DataFrame(X.toarray())
        data["target"] = 0.5 * (pd.DataFrame(y) + 1)

    elif dataset_name == "w7a":
        X, y = load_svmlight_file(f"data/raw/{dataset_name}/w7a")
        data = pd.DataFrame(X.toarray())
        data["target"] = 0.5 * (pd.DataFrame(y) + 1)

    elif dataset_name == "w8a":
        X, y = load_svmlight_file(f"data/raw/{dataset_name}/w8a")
        data = pd.DataFrame(X.toarray())
        data["target"] = 0.5 * (pd.DataFrame(y) + 1)

    elif dataset_name == "w9a":
        X, y = load_svmlight_file(f"data/raw/{dataset_name}/w9a")
        data = pd.DataFrame(X.toarray())
        data["target"] = 0.5 * (pd.DataFrame(y) + 1)

    else:
        raise NotImplementedError(f"{dataset_name}")
    return data


parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset", "-d", default=None, help="Which dataset to download.",
)
args = parser.parse_args()

# Load and process data
dataset_name = args.dataset
data = process(dataset_name)

# Basic assertions
target_unique = pd.unique(data["target"])
assert set(target_unique) == set([0,1]), f"Target is not 0-1 binary: {target_unique}."
assert all(ptypes.is_numeric_dtype(data[col]) for col in data.columns)

# Export data
output_folder = get_folder("data/processed")
data.to_csv(f"{output_folder}/{dataset_name}.csv", index=False)
print(f"-saved: {output_folder}/{dataset_name}.csv")
