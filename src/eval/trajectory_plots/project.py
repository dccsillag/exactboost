"""
Generate data for trajectory visualization plots.
"""

import os
from glob import glob
from argparse import ArgumentParser
import pickle

import numpy as np
from tqdm import tqdm
from umap import UMAP

from src.utils.model_utils import get_data


parser = ArgumentParser()
parser.add_argument('-m', '--metric',
                    choices=["auc", "ks", "pak"],
                    help="Which metric to use")
parser.add_argument('-d', '--dataset',
                    help="Dataset to train on")
parser.add_argument('-i', '--input',
                    default='eval/trajectories/{dataset}/exactboost_{metric}',
                    help="Path to read data from")
parser.add_argument('-o', '--output',
                    default='eval/viz/trajectories/{dataset}-exactboost_{metric}-{what}-{n_estimators}.npy',
                    help="Path to save figure to")
parser.add_argument('-S', '--seed',
                    type=int,
                    default=0,
                    help="Random seed to use for UMAP")
parser.add_argument('-e', '--n-estimators',
                    type=int,
                    required=True,
                    help="How many estimators to use (default = ALL)")
args = parser.parse_args()

np.random.seed(args.seed)


data = get_data(f"{args.dataset}_full_train")
n_features = data.shape[1] - 1

output_path = args.output.format(dataset=args.dataset,
                                 metric=args.metric,
                                 what='{what}',
                                 n_estimators='{n_estimators}')
if not os.path.exists(os.path.dirname(output_path)):
    os.makedirs(os.path.dirname(output_path))
input_path = args.input.format(dataset=args.dataset,
                               metric=args.metric)

N_ESTIMATORS = args.n_estimators
N_ROUNDS = 50

experiment_dirs = sorted(glob(os.path.join(input_path, 'experiment_*')))
n_experiments = len(experiment_dirs)

models_paramss = np.zeros((n_experiments*N_ROUNDS, N_ESTIMATORS*N_ROUNDS*4), dtype=np.float32)
metrics_train = np.zeros(n_experiments*N_ROUNDS)
metrics_test = np.zeros(n_experiments*N_ROUNDS)

for experiment_i, experiment_path in \
        enumerate(tqdm(experiment_dirs, desc="get params")):
    if not os.path.exists(os.path.join(experiment_path, 'eval')):
        raise RuntimeError("Missing eval directory for experiment: " + experiment_path)

    index = experiment_i*N_ROUNDS

    with open(os.path.join(experiment_path, "trained_model"), "rb") as file:
        trained_model = pickle.load(file)

    assert N_ROUNDS == trained_model.exactboost_kwargs["n_rounds"]
    assert N_ESTIMATORS <= trained_model.n_estimators

    for k, estimator in enumerate(trained_model.exactboost_models[:N_ESTIMATORS]):
        for i, stump in enumerate(estimator.params_history[:N_ROUNDS]):
            p = 4*N_ROUNDS*k + 4*i
            models_paramss[index+i:index+N_ROUNDS, p + 0] = stump.feature
            models_paramss[index+i:index+N_ROUNDS, p + 1] = stump.xi
            models_paramss[index+i:index+N_ROUNDS, p + 2] = stump.a
            models_paramss[index+i:index+N_ROUNDS, p + 3] = stump.b

    for i in range(N_ROUNDS):
        with open(os.path.join(experiment_path, 'eval', f"{args.metric}_train{i}-{N_ESTIMATORS}")) as file:
            metrics_train[index+i] = float(file.read())
        with open(os.path.join(experiment_path, 'eval', f"{args.metric}_test{i}-{N_ESTIMATORS}")) as file:
            metrics_test[index+i] = float(file.read())

print("=> UMAP")
reduced_points = UMAP(densmap=True, random_state=0, verbose=True, n_jobs=-1).fit_transform(models_paramss)

print("=> write to disk")
xs = reduced_points[:, 0]
ys = reduced_points[:, 1]

np.save(output_path.format(what='xs', n_estimators=args.n_estimators), xs)
np.save(output_path.format(what='ys', n_estimators=args.n_estimators), ys)

np.save(output_path.format(what='metrics_' + args.metric + "_train", n_estimators=args.n_estimators), metrics_train)
np.save(output_path.format(what='metrics_' + args.metric + "_test", n_estimators=args.n_estimators), metrics_test)
