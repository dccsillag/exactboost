from typing import Optional

from tqdm import tqdm
import numpy as np
from sklearn.base import clone
from sklearn.base import BaseEstimator as SciKitLearnModel
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import clone_model
from tensorflow.python.keras.engine.training import Model as TensorFlowModel
from tensorflow.random import set_seed

class Plugin:
    def __init__(self, base_model, metric, n_samples: int = 1000, thr_split: float = 0.2, random_state: Optional[int] = None):
        self.base_model = base_model
        self.metric = metric
        self.n_samples = n_samples
        self.thr_split = thr_split
        self.random_state = random_state

    def fit(self, X, y):
        X_est, X_thr, y_est, y_thr = train_test_split(X, y, test_size=self.thr_split, stratify=y, random_state=self.random_state)

        if isinstance(self.base_model, SciKitLearnModel):
            trained_model = clone(self.base_model).fit(X_est, y_est)
            predictions_thr = trained_model.predict_proba(X_thr)[:, 1]

        elif isinstance(self.base_model, TensorFlowModel):
            set_seed(0)
            trained_model = clone_model(self.base_model)
            trained_model.compile(optimizer="adam", loss="binary_crossentropy")
            trained_model.fit(
                X_est, y_est, epochs=30, batch_size=4
            )
            predictions_thr = trained_model.predict(X_thr).flatten()

        del trained_model

        thresholds_to_attempt = np.linspace(0, 1, self.n_samples)
        metric_per_threshold = []
        for threshold in tqdm(thresholds_to_attempt, desc="thresholds"):
            metric_per_threshold.append(self.metric(y_thr, np.where(predictions_thr >= threshold, 1, 0)))

        self.best_threshold = thresholds_to_attempt[np.argmax(metric_per_threshold)]

        if isinstance(self.base_model, SciKitLearnModel):
            self.trained_model = clone(self.base_model).fit(X, y)

        elif isinstance(self.base_model, TensorFlowModel):
            set_seed(0)
            self.trained_model = clone_model(self.base_model)
            self.trained_model.compile(optimizer="adam", loss="binary_crossentropy")
            self.trained_model.fit(
                X, y, epochs=30, batch_size=4
            )

    def predict(self, X):
        if isinstance(self.trained_model, SciKitLearnModel):
            return np.where(self.trained_model.predict_proba(X)[:, 1] >= self.best_threshold, 1, 0)
        elif isinstance(self.trained_model, TensorFlowModel):
            return np.where(self.trained_model.predict(X).flatten() >= self.best_threshold, 1, 0)
