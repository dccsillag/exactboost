"""
Generate data for trajectory visualization plots.
"""

import os
from glob import glob
from argparse import ArgumentParser
import pickle

from joblib import Parallel, delayed
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from src.utils.model_utils import get_X_y
from src.utils.eval_utils import ks_np, precision_at_k
from src.models.exactboost.model import ExactBoost

parser = ArgumentParser()
parser.add_argument('-m', '--metric',
                    choices=["auc", "ks", "pak"],
                    help="Which model to evaluate")
parser.add_argument('-d', '--dataset',
                    help="Dataset to train on")
parser.add_argument('-i', '-o', '--input', '--output',
                    dest='rootpath',
                    default='eval/trajectories/{dataset}/exactboost_{metric}',
                    help="Path to write output to")
parser.add_argument('-e', '--n-estimators',
                    type=int,
                    help="Number of estimators")
args = parser.parse_args()


root_path = args.rootpath.format(dataset=args.dataset, metric=args.metric)

X_full,  y_full  = get_X_y(f'{args.dataset}_full_train+test')
X_train, y_train = get_X_y(f'{args.dataset}_full_train')
X_test,  y_test  = get_X_y(f'{args.dataset}_full_test')

BETA = y_full.mean()

def predict(model, n_rounds, X, initial_score):
    score = initial_score.copy()
    X = X.copy()

    if isinstance(model, ExactBoost):
        for estimator in model.stumps[:args.n_estimators]:
            this_score = np.zeros(score.shape)
            for weak_learner in estimator[:n_rounds]:
                this_score = this_score + np.where(X[:, weak_learner.feature] <= weak_learner.xi, weak_learner.a, weak_learner.b)
            score = score + np.interp(this_score, (this_score.min(), this_score.max()), (0, 1))
        score = score / model.n_estimators
    else:
        for weak_learner in model.params_history[:n_rounds]:
            score = score + weak_learner(X)

    return np.interp(score, (score.min(), score.max()), (0, 1))


def eval_experiment(experiment_path):
    # Check for lock
    if os.path.exists(os.path.join(experiment_path, 'lock')):
        return

    np.random.seed(0)

    # Read model
    with open(os.path.join(experiment_path, 'trained_model'), "rb") as file:
        trained_model = pickle.load(file)

    N_ROUNDS = trained_model.n_rounds

    # Generate initial scores
    initial_score_train = np.random.uniform(size=y_train.shape)
    initial_score_test  = np.random.uniform(size=y_test.shape)

    # Create eval directory
    eval_path = os.path.join(experiment_path, 'eval')
    if not os.path.exists(eval_path):
        os.mkdir(eval_path)

    # Eval for each round
    for i in range(N_ROUNDS):
        # Get predictions
        assert not hasattr(trained_model, 'predict_proba')
        preds_train = predict(trained_model, i+1, X_train, initial_score_train)
        preds_test  = predict(trained_model, i+1, X_test,  initial_score_test)

        # Eval and save metrics
        # # KS
        with open(os.path.join(eval_path, 'ks_train%d-%d' % (i, args.n_estimators)), 'w') as file:
            file.write(repr(ks_np(y_train, preds_train)))
        with open(os.path.join(eval_path, 'ks_test%d-%d' % (i, args.n_estimators)), 'w') as file:
            file.write(repr(ks_np(y_test,  preds_test)))
        # # P@K
        with open(os.path.join(eval_path, 'pak_train%d-%d' % (i, args.n_estimators)), 'w') as file:
            file.write(repr(precision_at_k(y_train, preds_train, beta=BETA*0.5)))
        with open(os.path.join(eval_path, 'pak_test%d-%d' % (i, args.n_estimators)), 'w') as file:
            file.write(repr(precision_at_k(y_test, preds_test, beta=BETA*0.5)))
        # # AUC
        with open(os.path.join(eval_path, 'auc_train%d-%d' % (i, args.n_estimators)), 'w') as file:
            file.write(repr(roc_auc_score(y_train, preds_train)))
        with open(os.path.join(eval_path, 'auc_test%d-%d' % (i, args.n_estimators)), 'w') as file:
            file.write(repr(roc_auc_score(y_test,  preds_test)))

Parallel(n_jobs=-1)(delayed(eval_experiment)(experiment_path)
                    for experiment_path
                    in tqdm(sorted(glob(os.path.join(root_path, 'experiment_*')))))
