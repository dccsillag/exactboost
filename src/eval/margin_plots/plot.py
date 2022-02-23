import argparse

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", "-d")
parser.add_argument("--metric", "-m", default="ks")
parser.add_argument("--observation_subsampling", "-os", type=float, default=0.2)
parser.add_argument("--feature_subsampling", "-fs", type=float, default=0.2)
parser.add_argument("--n_rounds", "-r", type=int, default=50)
parser.add_argument("--n_estimators", "-e", type=int, default=250)
args = parser.parse_args()

metrics = {"auc": "AUC", "ks": "KS", "pak": "P@k"}

def do_plot(loss):
    df = pd.DataFrame(columns=["Margin", f"{metrics[METRIC]} loss", "Dataset"])
    df = df.append(
        pd.DataFrame(
            {
                "Margin": thetass[DATASET],
                f"{metrics[METRIC]} loss": loss[DATASET],
                "Dataset": [DATASET] * len(thetass[DATASET]),
            }
        )
    )

    plt.rcParams.update({"font.size": 13, "axes.labelsize": 15})
    fig, ax = plt.subplots()
    ax.yaxis.set_tick_params(pad=3)
    for lab in ax.yaxis.get_ticklabels():
        lab.set_verticalalignment("center")
    sns.lineplot(x="Margin", y=f"{metrics[METRIC]} loss", hue="Dataset", data=df, palette=palette, ax=ax)
    ax.set_title(f"{DATASET} -os {OBS_SUBSAMPLING} -fs {FEAT_SUBSAMPLING} -r {N_ROUNDS} -e {N_TREES}")

    return fig

DATASET = args.dataset
SPLIT_SEED = 0

N_SAMPLES = 256
N_JOBS = 64

N_TREES = args.n_estimators
N_ROUNDS = args.n_rounds
OBS_SUBSAMPLING = args.observation_subsampling
FEAT_SUBSAMPLING = args.feature_subsampling
METRIC = args.metric

thetass = {}
train_loss = {}
test_loss = {}

train_data = np.load(
    f"eval/margin_plots/{METRIC}_margin-{DATASET}-"
    f"train-os_{OBS_SUBSAMPLING}-fs_{FEAT_SUBSAMPLING}-r_{N_ROUNDS}-"
    f"estimators_{N_TREES}-split_{SPLIT_SEED}.npy",
    allow_pickle=True,
)[()]
test_data = np.load(
    f"eval/margin_plots/{METRIC}_margin-{DATASET}-"
    f"test-os_{OBS_SUBSAMPLING}-fs_{FEAT_SUBSAMPLING}-r_{N_ROUNDS}-"
    f"estimators_{N_TREES}-split_{SPLIT_SEED}.npy",
    allow_pickle=True,
)[()]

thetass[DATASET] = next(iter(train_data.keys()))
train_loss[DATASET] = np.array(train_data[thetass[DATASET]])
test_loss[DATASET] = np.array(test_data[thetass[DATASET]])

matplotlib.use("pgf")

palette = [(r / 255, g / 255, b / 255) for (r, g, b) in [(0, 48, 73)]]

do_plot(train_loss).savefig(
    f"eval/margin_plots/{METRIC}_margin-{DATASET}-train-"
    f"os_{OBS_SUBSAMPLING}-fs_{FEAT_SUBSAMPLING}-r_{N_ROUNDS}-estimators_{N_TREES}"
    f"-split_{SPLIT_SEED}.pgf",
    bbox_inches="tight",
)
do_plot(test_loss).savefig(
    f"eval/margin_plots/{METRIC}_margin-{DATASET}-test-"
    f"os_{OBS_SUBSAMPLING}-fs_{FEAT_SUBSAMPLING}-r_{N_ROUNDS}-estimators_{N_TREES}"
    f"-split_{SPLIT_SEED}.pgf",
    bbox_inches="tight",
)
