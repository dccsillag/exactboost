import gc
import os
import pickle

import numpy as np
import pandas as pd
import sklearn.ensemble
import tensorflow as tf
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier

from src.models.dmks.model import imo, pestimate
from src.models.exactboost.model import ExactBoost, AUC, KS, PaK
from src.models.plugin.model import Plugin
from src.models.rankboost.model import RankBoost
from src.models.svmperf.model import SVMPerf
from src.utils.general_utils import save_pandas_to_hdf, get_folder
from src.utils.model_utils import get_X_y

def run_models(
    i,
    train_index,
    test_index,
    CV_OR_VAL,
    DATASET,
    MODEL_NAME,
    TEST_SET,
    N_SPLITS,
    N_TREES,
    N_BINS,
    N_ROUNDS,
    MARGIN_THETA,
    N_JOBS_MODEL,
    BETA,
    K,
    OBSERVATION_SUBSAMPLING,
    FEATURE_SUBSAMPLING,
    RECREATE_IF_EXISTING,
):
    FULL_DATA = f"{DATASET}_full_train+test"

    output_folder = get_folder(f"eval/boxplots-{CV_OR_VAL}/models")
    model_folder = get_folder(f"{output_folder}/{MODEL_NAME}/{DATASET}/trained_models/")
    preds_folder = get_folder(f"{output_folder}/{MODEL_NAME}/{DATASET}/preds")

    model_path = os.path.join(
        model_folder, f"{MODEL_NAME}-{CV_OR_VAL}-{TEST_SET}-{i}_{N_SPLITS}.pickle"
    )
    preds_path = (
        f"{preds_folder}/" f"{MODEL_NAME}-{CV_OR_VAL}-{TEST_SET}-{i}_{N_SPLITS}.h5"
    )

    if os.path.exists(preds_path) and not RECREATE_IF_EXISTING:
        print(f"Predictions exist at {preds_path}; skipping.")
        return

    # Train
    X, y = get_X_y(FULL_DATA, subset_index=train_index, seed=0)

    if MODEL_NAME == "adaboost":
        model = sklearn.ensemble.AdaBoostClassifier(random_state=0)
        model.fit(X, y)
        pickle.dump(model, open(model_path, "wb"))
    elif MODEL_NAME == "dmks":
        X = X + np.finfo(float).eps
        init_model = LogisticRegression(random_state=0, n_jobs=N_JOBS_MODEL)
        init_model.fit(X, y)
        beta_zero = init_model.coef_[0] / np.linalg.norm(init_model.coef_[0])
        model = imo(X, y, weight=np.ones(X.shape[1]), beta_start=beta_zero, lamb=0)[0]
        model_path = f"{'.'.join(model_path.split('.')[:-1])}.model"
        np.savez(f"{model_path}", model, y, X @ model)
        os.rename(f"{model_path}.npz", f"{model_path}")
    elif MODEL_NAME == "knn":
        model = KNeighborsClassifier(n_jobs=N_JOBS_MODEL)
        model.fit(X, y)
        pickle.dump(model, open(model_path, "wb"))
    elif MODEL_NAME == "logistic":
        model = LogisticRegression(random_state=0, n_jobs=N_JOBS_MODEL)
        model.fit(X, y)
        pickle.dump(model, open(model_path, "wb"))
    elif MODEL_NAME == "neural_network":
        tf.random.set_seed(0)
        model = tf.keras.models.Sequential(
            [
                tf.keras.layers.Dense(26, activation="relu"),
                tf.keras.layers.Dense(12, activation="relu"),
                tf.keras.layers.Dense(12, activation="relu"),
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ]
        )
        model.compile(optimizer="adam", loss="binary_crossentropy")
        model.fit(X, y, epochs=30, batch_size=4)
        nn_folder = get_folder(model_path.split(".")[0])
        model.save(nn_folder)
    elif MODEL_NAME == "plugin_logistic":
        logistic_regression = LogisticRegression(random_state=0, n_jobs=N_JOBS_MODEL)
        model = Plugin(base_model=logistic_regression, metric=roc_auc_score, random_state=0)
        model.fit(X, y)
        pickle.dump(model, open(model_path, "wb"))
    elif MODEL_NAME == "plugin_neural_network":
        tf.random.set_seed(0)
        neural_network = tf.keras.models.Sequential(
            [
                tf.keras.layers.Dense(26, activation="relu"),
                tf.keras.layers.Dense(12, activation="relu"),
                tf.keras.layers.Dense(12, activation="relu"),
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ]
        )
        neural_network.compile(optimizer="adam", loss="binary_crossentropy")
        model = Plugin(base_model=neural_network, metric=roc_auc_score, random_state=0)
        model.fit(X, y)
    elif MODEL_NAME == "random_forest":
        model = sklearn.ensemble.RandomForestClassifier(
            random_state=0, n_jobs=N_JOBS_MODEL
        )
        model.fit(X, y)
        pickle.dump(model, open(model_path, "wb"))
    elif MODEL_NAME == "svmperf_auc":
        model = SVMPerf(loss="auc")
        model.fit(X, y)
    elif MODEL_NAME == "svmperf_pak":
        model = SVMPerf(loss="precision_at_k", beta=BETA)
        model.fit(X, y)
    elif MODEL_NAME == "top_push":
        from julia import Julia

        # Use a custom system image, with PyCall (1) and ClassificationOnTop (2) compiled in.
        # (1) is a workaround needed due to python being installed through Conda.
        # (2) greatly reduces initialization time.
        jl = Julia(
            sysimage="setup/julia-1.5.3/custom_sysimage.so", compiled_modules=False
        )
        from julia import ClassificationOnTop

        alg = ClassificationOnTop.TopPush()
        w_ini = np.zeros(X.shape[1])
        model, _, _ = ClassificationOnTop.solver(alg, (X, y), w=w_ini)
    elif MODEL_NAME == "xgboost":
        model = xgb.XGBClassifier(seed=0, n_jobs=N_JOBS_MODEL)
        model.fit(X, y)
        pickle.dump(model, open(model_path, "wb"))
    elif MODEL_NAME == "exactboost_ks":
        metric = KS()
        model = ExactBoost(metric)
        model.fit(X, y)
        pickle.dump(model, open(model_path, "wb"))
    elif MODEL_NAME == "exactboost_pak":
        metric = PaK(K)
        model = ExactBoost(metric)
        model.fit(X, y)
        pickle.dump(model, open(model_path, "wb"))
    elif MODEL_NAME == "exactboost_auc":
        metric = AUC()
        model = ExactBoost(metric)
        model.fit(X, y)
        pickle.dump(model, open(model_path, "wb"))
    elif MODEL_NAME == "rankboost":
        np.random.seed(0)
        model = RankBoost(n_rounds=100)
        model.fit(X, y)
        with open(model_path, 'wb') as file:
            pickle.dump(model, file)
    else:
        raise ValueError(f"Model {MODEL_NAME} not currently included for training.")

    # Predict
    X, y = get_X_y(FULL_DATA, subset_index=test_index, seed=0)

    if MODEL_NAME in ["exactboost_ks", "exactboost_pak", "exactboost_auc"]:
        y_pred = model.predict(X)
    elif MODEL_NAME == "rankboost":
        y_pred = model.predict(X)
    elif MODEL_NAME == "dmks":
        model = np.load(f"{model_path}")
        y_pred = pestimate(model["arr_2"], model["arr_1"], X @ model["arr_0"])
    elif MODEL_NAME == "neural_network":
        y_pred = model.predict(X).flatten()
    elif MODEL_NAME == "top_push":
        y_pred = X @ model.w + model.bias
    elif MODEL_NAME == "plugin_logistic":
        y_pred = model.predict(X)
    elif MODEL_NAME == "plugin_neural_network":
        y_pred = model.predict(X)
    elif MODEL_NAME == "svmperf_auc":
        y_pred = model.predict(X)
    elif MODEL_NAME == "svmperf_pak":
        y_pred = model.predict(X)
    elif MODEL_NAME in [
        "adaboost",
        "knn",
        "logistic",
        "random_forest",
        "xgboost",
    ]:
        y_pred = [model_class[1] for model_class in model.predict_proba(X)]
    else:
        raise ValueError(f"Model {MODEL_NAME} not currently included for predicting.")

    preds_df = pd.DataFrame(columns=["preds", "target"])
    preds_df["preds"] = y_pred
    preds_df["target"] = y
    save_pandas_to_hdf(preds_df, preds_path)
    del X, y, model, y_pred, preds_df
    gc.collect()


def combine_predictions(
    i,
    train_index,
    test_index,
    CV_OR_VAL,
    DATASET,
    MODELS,
    TEST_SET,
    N_SPLITS,
    PREDS_ONLY,
):
    FULL_DATA = f"{DATASET}_full_train+test"

    output_folder = get_folder(
        f"eval/boxplots-{CV_OR_VAL}/models/combined_preds/{DATASET}"
    )
    output_filename = f"combined_preds-{CV_OR_VAL}-{TEST_SET}-{i}_{N_SPLITS}.h5"

    combined_preds = pd.DataFrame()

    for model in MODELS:
        preds_folder = get_folder(
            f"eval/boxplots-{CV_OR_VAL}/models/{model}/{DATASET}/preds"
        )
        preds_path = (
            f"{preds_folder}/" f"{model}-{CV_OR_VAL}-{TEST_SET}-{i}_{N_SPLITS}.h5"
        )

        preds = pd.read_hdf(preds_path)
        preds = preds.rename({"preds": model}, axis=1)
        preds = preds[model]
        combined_preds = pd.concat([combined_preds, preds], axis=1)

    X, y = get_X_y(FULL_DATA, subset_index=test_index, seed=0)
    combined_preds["target"] = y

    if not PREDS_ONLY:
        combined_preds = pd.concat([combined_preds, pd.DataFrame(X)], axis=1)

    save_pandas_to_hdf(combined_preds, f"{output_folder}/{output_filename}")


def run_ensembler(
    i,
    CV_OR_VAL,
    DATASET,
    ENSEMBLER,
    N_SPLITS,
    N_TREES,
    N_BINS,
    N_ROUNDS,
    MARGIN_THETA,
    N_JOBS_MODEL,
    BETA,
    K,
    OBSERVATION_SUBSAMPLING,
    FEATURE_SUBSAMPLING,
    RECREATE_IF_EXISTING,
):
    combined_folder = get_folder(
        f"eval/boxplots-{CV_OR_VAL}/models/combined_preds/{DATASET}"
    )
    output_base_folder = get_folder(
        f"eval/boxplots-{CV_OR_VAL}/models/{ENSEMBLER}/{DATASET}"
    )
    output_model_folder = get_folder(f"{output_base_folder}/trained_models")
    output_preds_folder = get_folder(f"{output_base_folder}/preds")
    output_model_file = (
        f"{output_model_folder}/{ENSEMBLER}-{CV_OR_VAL}-test-{i}_{N_SPLITS}.pickle"
    )
    output_preds_file = (
        f"{output_preds_folder}/{ENSEMBLER}-{CV_OR_VAL}-test-{i}_{N_SPLITS}.h5"
    )
    if os.path.exists(output_preds_file) and not RECREATE_IF_EXISTING:
        print(f"Predictions exist at {output_preds_file}; skipping.")
        return

    train_data = pd.read_hdf(
        f"{combined_folder}/combined_preds-{CV_OR_VAL}-ens-{i}_{N_SPLITS}.h5"
    )
    test_data = pd.read_hdf(
        f"{combined_folder}/combined_preds-{CV_OR_VAL}-test-{i}_{N_SPLITS}.h5"
    )

    y_train = np.ravel(train_data[["target"]])
    X_train = train_data.drop("target", axis=1).to_numpy()
    initial_score_train = np.zeros(X_train.shape[0])

    y_test = np.ravel(test_data[["target"]])
    X_test = test_data.drop("target", axis=1).to_numpy()
    initial_score_test = np.zeros(X_test.shape[0])

    if ENSEMBLER == "exactboost_ks_ensembler":
        metric = KS()
        model = ExactBoost(metric)
        model.fit(X_train, y_train)
        pickle.dump(model, open(output_model_file, "wb"))
        y_pred = model.predict(X_test)
    elif ENSEMBLER == "exactboost_pak_ensembler":
        metric = PaK(K)
        model = ExactBoost(metric)
        model.fit(X_train, y_train)
        pickle.dump(model, open(output_model_file, "wb"))
        y_pred = model.predict(X_test)
    elif ENSEMBLER == "exactboost_auc_ensembler":
        metric = AUC()
        model = ExactBoost(metric)
        model.fit(X_train, y_train)
        pickle.dump(model, open(output_model_file, "wb"))
        y_pred = model.predict(X_test)
    elif ENSEMBLER == "rankboost_ensembler":
        np.random.seed(0)
        model = RankBoost(n_rounds=100)
        model.fit(X_train, y_train)
        with open(output_model_file, 'wb') as file:
            pickle.dump(model, file)
        y_pred = model.predict(X_test)
    elif ENSEMBLER == "adaboost_ensembler":
        model = sklearn.ensemble.AdaBoostClassifier(random_state=0)
        model.fit(X_train, y_train)
        pickle.dump(model, open(output_model_file, "wb"))
        y_pred = [model_class[1] for model_class in model.predict_proba(X_test)]
    elif ENSEMBLER == "dmks_ensembler":
        X_train = X_train + np.finfo(float).eps
        init_model = LogisticRegression(random_state=0, n_jobs=N_JOBS_MODEL)
        init_model.fit(X_train, y_train)
        beta_zero = init_model.coef_[0] / np.linalg.norm(init_model.coef_[0])
        model = imo(X_train, y_train, weight=np.ones(X_train.shape[1]), beta_start=beta_zero, lamb=0)[0]
        model_path = f"{'.'.join(output_model_file.split('.')[:-1])}.model"
        np.savez(f"{model_path}", model, y_train, X_train @ model)
        os.rename(f"{model_path}.npz", f"{model_path}")
        model = np.load(f"{model_path}")
        y_pred = pestimate(model["arr_2"], model["arr_1"], X_test @ model["arr_0"])
    elif ENSEMBLER == "knn_ensembler":
        model = KNeighborsClassifier(n_jobs=N_JOBS_MODEL)
        model.fit(X_train, y_train)
        pickle.dump(model, open(output_model_file, "wb"))
        y_pred = [model_class[1] for model_class in model.predict_proba(X_test)]
    elif ENSEMBLER == "logistic_ensembler":
        model = sklearn.linear_model.LogisticRegression(random_state=0)
        model.fit(X_train, y_train)
        pickle.dump(model, open(output_model_file, "wb"))
        y_pred = [model_class[1] for model_class in model.predict_proba(X_test)]
    elif ENSEMBLER == "neural_network_ensembler":
        tf.random.set_seed(0)
        model = tf.keras.models.Sequential(
            [
                tf.keras.layers.Dense(26, activation="relu"),
                tf.keras.layers.Dense(12, activation="relu"),
                tf.keras.layers.Dense(12, activation="relu"),
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ]
        )
        model.compile(optimizer="adam", loss="binary_crossentropy")
        model.fit(X_train, y_train, epochs=30, batch_size=4)
        nn_folder = get_folder(output_model_file.split(".")[0])
        model.save(nn_folder)
        y_pred = model.predict(X_test).flatten()
    elif ENSEMBLER == "plugin_logistic_ensembler":
        logistic_regression = LogisticRegression(random_state=0, n_jobs=N_JOBS_MODEL)
        model = Plugin(base_model=logistic_regression, metric=roc_auc_score, random_state=0)
        model.fit(X_train, y_train)
        pickle.dump(model, open(output_model_file, "wb"))
        y_pred = model.predict(X_test)
    elif ENSEMBLER == "plugin_neural_network_ensembler":
        tf.random.set_seed(0)
        neural_network = tf.keras.models.Sequential(
            [
                tf.keras.layers.Dense(26, activation="relu"),
                tf.keras.layers.Dense(12, activation="relu"),
                tf.keras.layers.Dense(12, activation="relu"),
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ]
        )
        neural_network.compile(optimizer="adam", loss="binary_crossentropy")
        model = Plugin(base_model=neural_network, metric=roc_auc_score, random_state=0)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    elif ENSEMBLER == "random_forest_ensembler":
        model = sklearn.ensemble.RandomForestClassifier(
            random_state=0, n_jobs=N_JOBS_MODEL
        )
        model.fit(X_train, y_train)
        pickle.dump(model, open(output_model_file, "wb"))
        y_pred = model.predict(X_test)
    elif ENSEMBLER == "svmperf_auc_ensembler":
        model = SVMPerf(loss="auc")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    elif ENSEMBLER == "svmperf_pak_ensembler":
        model = SVMPerf(loss="precision_at_k", beta=BETA)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    elif ENSEMBLER == "top_push_ensembler":
        from julia import Julia

        # Use a custom system image, with PyCall (1) and ClassificationOnTop (2) compiled in.
        # (1) is a workaround needed due to python being installed through Conda.
        # (2) greatly reduces initialization time.
        jl = Julia(
            sysimage="setup/julia-1.5.3/custom_sysimage.so", compiled_modules=False
        )
        from julia import ClassificationOnTop

        alg = ClassificationOnTop.TopPush()
        w_ini = np.zeros(X_train.shape[1])
        model, _, _ = ClassificationOnTop.solver(alg, (X_train, y_train), w=w_ini)
        y_pred = X_test @ model.w + model.bias
    elif ENSEMBLER == "xgboost_ensembler":
        model = xgb.XGBClassifier(seed=0, n_jobs=N_JOBS_MODEL)
        model.fit(X_train, y_train)
        pickle.dump(model, open(output_model_file, "wb"))
        y_pred = model.predict(X_test)
    else:
        raise RuntimeError(f"Model {ENSEMBLER} cannot be used as an ensembler.")

    preds_df = pd.DataFrame(columns=["preds", "target"])
    preds_df["preds"] = y_pred
    preds_df["target"] = y_test
    save_pandas_to_hdf(preds_df, output_preds_file)
