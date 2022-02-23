import argparse

from joblib import Parallel, delayed

from tqdm import tqdm

from src.eval.boxplots.utils import run_ensembler
from src.utils.model_utils import get_X_y

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    "-d",
    default=None,
    help="Which dataset to use for training and prediction.",
)
parser.add_argument(
    "--ensembler",
    "-e",
    default="exactboost_ks",
    help="Which model to use for ensembling.",
)
parser.add_argument(
    "--validation",
    "-val",
    action="store_true",
    help="If True, repeatedly subsample data to generate boxplot train and test folds; "
    "if False, use cross-validation (i.e., use independent folds for train and test).",
)
parser.add_argument(
    "--n_jobs",
    "-j",
    type=int,
    default=1,
    help="How many jobs to launch in parallel, one for each CV fold.",
)
parser.add_argument(
    "--n_folds",
    "-f",
    type=int,
    default=5,
    help="Number of folds to use in cross-validation.",
)
parser.add_argument(
    "--n_seeds",
    "-s",
    type=int,
    default=10,
    help="Number of seeds for resampling; "
    "only used in validation mode (i.e., CV_OR_VAL == 'VAL').",
)
parser.add_argument(
    "--n_trees",
    "-t",
    type=int,
    default=150,
    help="Number of trees to average in ExactBoost.",
)
parser.add_argument(
    "--n_bins",
    "-b",
    type=int,
    default=500,
    help="Number of bins to use in data to accelerate ExactBoost.",
)
parser.add_argument(
    "--n_rounds",
    "-r",
    type=int,
    default=150,
    help="Number of rounds of ExactBoost training.",
)
parser.add_argument(
    "--margin_theta",
    "-mt",
    type=float,
    default=0.05,
    help="Margin",
)
parser.add_argument(
    "--n_jobs_model",
    "-jm",
    type=int,
    default=1,
    help="How many jobs to launch in parallel for models.",
)
parser.add_argument(
    "--k",
    "-k",
    default=None,
    help="Precision of k topmost observations are to used when evaluating "
    "with precision at k metric (pak).",
)
parser.add_argument(
    "--beta",
    "-B",
    type=float,
    default=None,
    help="Alternative way of specifying k, where k = num_obs*beta; "
    "if k is specified, it has precedence.",
)
parser.add_argument(
    "--observation_subsampling",
    "-os",
    type=float,
    default=0.5,
    help="Fraction of observations to sample in each ExactBoost round.",
)
parser.add_argument(
    "--feature_subsampling",
    "-fs",
    type=float,
    default=1,
    help="Fraction of features to sample in each ExactBoost round.",
)
parser.add_argument(
    "--recreate_if_existing",
    "-rie",
    action="store_true",
    help="If True, recreates model forecasts even if forecasts already exist.",
)
args = parser.parse_args()

CV_OR_VAL = "val" if args.validation else "cv"
DATASET = args.dataset
ENSEMBLER = (
    args.ensembler
    if args.ensembler.endswith("_ensembler")
    else args.ensembler + "_ensembler"
)
N_JOBS = args.n_jobs
N_SPLITS = args.n_seeds if args.validation else args.n_folds

N_TREES = args.n_trees
N_BINS = args.n_bins
N_ROUNDS = args.n_rounds
MARGIN_THETA = args.margin_theta
N_JOBS_MODEL = args.n_jobs_model
BETA = args.beta
K = args.k
OBSERVATION_SUBSAMPLING = args.observation_subsampling
FEATURE_SUBSAMPLING = args.feature_subsampling
RECREATE_IF_EXISTING = args.recreate_if_existing

FULL_DATA = f"{DATASET}_full_train+test"
X, y = get_X_y(FULL_DATA, seed=0)

if BETA is None and K is None:
    BETA = y.mean()

MODEL_PARAMS = [
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
]

model_inputs = [(i, *MODEL_PARAMS) for i in range(N_SPLITS)]


Parallel(n_jobs=N_JOBS)(
    delayed(run_ensembler)(*model_input) for model_input in tqdm(model_inputs)
)
