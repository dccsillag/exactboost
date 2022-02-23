import os
import uuid

import numpy as np


class SVMPerf:
    def __init__(self, loss, beta=None):
        self.loss = loss
        self.beta = beta

    def _dump_data(self, X, y, fname):
        with open(fname, 'w') as file:
            for i in range(X.shape[0]):
                print(('+1' if y[i] == 1 else '-1') + ' '
                      + ' '.join([f"{j+1}:{X[i,j]}" for j in range(X.shape[1])]),
                      file=file)

    def _mod_path(self, path):
        return path + str(uuid.uuid4())

    def fit(self, X, y):
        train_data_path = self._mod_path("svmperf-fit-data")
        self.model_path = self._mod_path("svmperf-model")

        # "Make sure your data is properly scaled.
        #  Normalize all feature vectors to Euclidian length 1"
        #    - https://www.cs.cornell.edu/people/tj/svm_light/svm_light_faq.html

        # calculate euclidean norm of features from training data
        self.features_norms = np.sqrt(np.sum(X ** 2, axis=0))
        # if norm is too close to zero (numerically equal to zero) ignore in order to avoid occurrences of NaN
        self.features_norms[self.features_norms == 0] = 1

        # ensure that X has type float
        X = X.astype(float)

        # normalize X to unit norm
        X /= self.features_norms

        self._dump_data(X, y, train_data_path)

        # running svm_perf with default choice of structural learning algorithm
        # for any loss that is not error rate results in the following message:
        #   "The custom algorithm can only optimize errorrate as the loss function.
        #      Please use algorithm '-w 3' instead."
        # so we keep the default option of '-w 9' for error rate loss only and use
        # '-w 3' for all other losses, as instructed
        struct_alg = 9 if self.loss == "error_rate" else 3

        # precision at k loss should be trained with specified fraction of positive
        # examples, our parameter 'beta', passed to svmperf via '--p' (posratio)
        extra_args = f"--p {self.beta}" if self.loss == "precision_at_k" and self.beta is not None else ""

        loss_options = {
            "zero_one_loss": 0,
            "f1": 1,
            "error_rate": 2,
            "precision_recall_breakeven": 3,
            "precision_at_k": 4,
            "recall_at_k": 5,
            "auc": 10,
        }

        loss = loss_options[self.loss]

        os.system(f"svm_perf_learn -c 0.01 -w {struct_alg} -l {loss} {extra_args} {train_data_path} {self.model_path}")
        os.remove(train_data_path)

    def predict(self, X):
        predict_data_path = self._mod_path("svmperf-predict-data")
        preds_file_path = self._mod_path("svmperf-preds")

        # ensure that X has type float
        X = X.astype(float)

        # normalize X using norms calculated on training data
        X /= self.features_norms

        self._dump_data(X, np.ones(X.shape[0]), predict_data_path)

        os.system(f"svm_perf_classify {predict_data_path} {self.model_path} {preds_file_path}")

        with open(preds_file_path, 'r') as file:
            preds = np.array([float(x) for x in file.read().split('\n')[:-1]])
        preds = np.interp(preds, (min(preds), max(preds)), (0, 1))

        assert preds.shape[0] == X.shape[0]

        for file in [predict_data_path, self.model_path, preds_file_path]:
            os.remove(file)

        return preds
