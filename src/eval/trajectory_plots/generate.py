"""
Generate data for trajectory and landscape visualization plots.
"""

from math import ceil
import itertools
import os
import pickle
import shutil
from argparse import ArgumentParser

import numpy as np
from tqdm import tqdm

from src.utils.model_utils import get_X_y
from src.models.exactboost.model import ExactBoost, AUC, KS, PaK


parser = ArgumentParser()
parser.add_argument('-m', '--metric',
                    choices=["auc", "ks", "pak"],
                    required=True,
                    help="Which model to train")
parser.add_argument("-d", "--dataset",
                    required=True,
                    help="Dataset to train on")
parser.add_argument('-o', '--output',
                    default='eval/trajectories/{dataset}/exactboost_{metric}',
                    help="Path to write output to")
parser.add_argument('-s', '--seed',
                    type=int,
                    default=0,
                    help="Random seed to use")
args = parser.parse_args()


X_full,  y_full  = get_X_y(f'{args.dataset}_full_train+test')
X_train, y_train = get_X_y(f'{args.dataset}_full_train')
X_test,  y_test  = get_X_y(f'{args.dataset}_full_test')

BETA = y_full.mean()
K = ceil(len(y_full)*BETA)


output_path = args.output.format(dataset=args.dataset,
                                 metric=args.metric)
if not os.path.exists(output_path):
    os.makedirs(output_path)

HYPERPARAMS = {
    "n_rounds": [50],
    "n_estimators": [250],
    "margin": np.linspace(0, 0.1, 5),
    "observation_subsampling": np.linspace(0, 1, 10),
    # "acceptance_rate": np.linspace(0, 1, 10),
}
hyperparams = [dict(x)
               for x in itertools.product(*[[(k, v) for v in vs]
                                            for k, vs in HYPERPARAMS.items()])]

for i, params in enumerate(tqdm(hyperparams, desc="hyperparams")):
    np.random.seed(args.seed)

    # Init
    experiment_path = os.path.join(output_path, 'experiment_%d' % i)
    if os.path.exists(experiment_path):
        shutil.rmtree(experiment_path)
    os.makedirs(experiment_path)

    # Create lock
    lock_path = os.path.join(experiment_path, 'lock')
    open(lock_path, 'w').close()

    # Train model
    metric = {
        "auc": AUC(),
        "ks": KS(),
        "pak": PaK(K),
    }[args.metric]
    trained_model = ExactBoost(metric=metric, **params)
    trained_model.fit(X_train, y_train)

    # Save trained model
    with open(os.path.join(experiment_path, 'trained_model'), "wb") as file:
        pickle.dump(trained_model, file)

    # Remove lock
    os.remove(lock_path)
