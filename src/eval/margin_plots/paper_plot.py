import argparse
import subprocess as sp

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter

from src.utils.general_utils import get_folder

parser = argparse.ArgumentParser()
parser.add_argument("--observation_subsampling", "-os", type=float, default=0.2)
parser.add_argument("--feature_subsampling", "-fs", type=float, default=0.2)
parser.add_argument("--n_rounds", "-r", type=int, default=50)
parser.add_argument("--n_estimators", "-e", type=int, default=250)
parser.add_argument("--split_seed", type=int, default=0)
args = parser.parse_args()

OBS_SUBSAMPLING = args.observation_subsampling
FEAT_SUBSAMPLING = args.feature_subsampling
N_ROUNDS = args.n_rounds
N_ESTIMATORS = args.n_estimators
SPLIT_SEED = args.split_seed

configs = [
    ("svmguide1", "auc"),
    ("gmsc", "ks"),
    ("splice", "pak")
]

n_plots = len(configs)

metrics_dict = {"auc": "AUC", "ks": "KS", "pak": "P@k"}

subplots = [None] * n_plots

for i in range(n_plots):
    subplots[i] = np.load(
        f"eval/margin_plots/{configs[i][1]}_margin-{configs[i][0]}-"
        f"test-os_{OBS_SUBSAMPLING}-fs_{FEAT_SUBSAMPLING}-"
        f"r_{N_ROUNDS}-estimators_{N_ESTIMATORS}-split_{SPLIT_SEED}.npy",
        allow_pickle=True,
    )[()]

def yticks(y_values):
    y_values = np.asarray(y_values)
    ymin = y_values.min()
    ymax = y_values.max()
    mid = (ymin + ymax) / 2
    ymin += (mid - ymin) / 4
    ymax -= (ymax - mid) / 4
    return [ymin, ymax]

matplotlib.use("pgf")

plt.rcParams.update({"font.size": 13, "axes.labelsize": 15})

fig, axs = plt.subplots(n_plots, 1, figsize=(6, 4.5), sharex=True)

for i in range(n_plots):
    axs[i].plot(
        list(subplots[i].keys())[0],
        list(subplots[i].values())[0],
        linewidth=1.2,
        color="#000075",
        label=f"{configs[i][0]}"
    )
    axs[i].set_ylabel(f"{metrics_dict[configs[i][1]]}")
    axs[i].set_yticks(yticks(list(subplots[i].values())))
    axs[i].axvline(0.05, linewidth=1, linestyle="dashed", color="black")
    axs[i].yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    axs[i].yaxis.set_tick_params(pad=3)
    for lab in axs[i].yaxis.get_ticklabels():
        lab.set_verticalalignment("center")

axs[0].xaxis.set_visible(False)
axs[1].xaxis.set_visible(False)
axs[2].set_xlabel("Margin")

fig.tight_layout()

plt.subplots_adjust(hspace=0.05, wspace=0)

path = get_folder("eval/margin_plots")

plt.savefig(
    f"{path}/margin_plots.pgf",
    bbox_inches="tight",
)

# Remove font definitions from generated pgf file. This ensures the figure
# inherits the parent LaTeX document font.
sp.call(["sed", "-i", r"s/\\..family//g", f"{path}/margin_plots.pgf"])
