
import numpy as np
import pandas as pd


def get_data(
    data_id,
    seed=0,
):
    """Returns data given data_id identifier. For example, it might return the result
    of splits of the labeled data into train or test sets, or splits into train, val,
    ens and test sets.

    Args:
        data_id (str): Identifier for dataset, in the format dataset_name, dataset_type,
            dataset_fold (e.g., diabetes_full_train, diabetes_full_test;
            diabetes_std_train, diabetes_std_train+ens, diabetes_std_val+ens,
            diabetes_std_ens, diabetes_std_test).
        seed (int): Seed for selecting the train, val, ens and test subsets of dataset.

    Returns:
        pd.DataFrame: Subset of labeled data specified according to data identifier.

    Raises:
        AssertionError: If datatype or dataset provided via data_id is unrecognized; in
            that case there is no proper specification for how to subset labeled data.
    """

    dataset_name, dataset_type, dataset_fold = data_id.split("_")
    assert dataset_type in ["std", "full"]
    assert set(dataset_fold.split("+")).issubset(["train", "ens", "val", "test"])

    # data = dataset_read(dataset_path)
    data = pd.read_csv(f"data/processed/{dataset_name}.csv")

    if dataset_type == "full":
        train_proportion = 0.75

        train_data = data.sample(frac=train_proportion, random_state=seed)
        test_data = data.drop(train_data.index)

        split_data = {
            "train": train_data,
            "test": test_data,
        }
    elif dataset_type == "std":
        train_proportion = 0.6
        test_proportion = 0.25

        test_data = data.drop(
            data.sample(frac=1 - test_proportion, random_state=seed).index
        )
        train_data = data.drop(test_data.index).sample(
            frac=train_proportion, random_state=seed
        )
        val_data = data.drop(train_data.index.union(test_data.index)).sample(
            frac=0.5, random_state=seed
        )
        ens_data = data.drop(
            train_data.index.union(test_data.index).union(val_data.index)
        )

        split_data = {
            "train": train_data,
            "val": val_data,
            "test": test_data,
            "ens": ens_data,
        }

    return pd.concat(split_data[x] for x in dataset_fold.split("+"))


def get_X_y(data_id, seed=0, subset_index=None):
    """Given dataset id and features, returns a random subset

    Args:
        data_id (str): Identifier for dataset, in the format dataset_name, dataset_type,
            dataset_fold (e.g., diabetes_full_train, diabetes_full_test;
            diabetes_std_train, diabetes_std_train+ens, diabetes_std_val+ens,
            diabetes_std_ens, diabetes_std_test).
        seed (int): Seed for selecting the train, val, ens and test subsets of dataset.
        subset_index (np.array): Array with fixed indices to use in subsetting data.

    Returns:
        tuple: feature matrix X and label vector y, both Numpy arrays.

    """
    data = get_data(data_id, seed=seed)
    data = data.fillna(0) + 0  # Fill NA and convert booleans to integers.
    if subset_index is not None:
        data = data.iloc[subset_index]
    X = data.drop(["target"], axis=1)
    y = np.ravel(data[["target"]])
    X = X.to_numpy()
    return X, y
