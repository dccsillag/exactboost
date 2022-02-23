"""
Generate data for trajectory visualizatoin plots.
"""

from argparse import ArgumentParser

from tqdm import tqdm
import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.collections import LineCollection
import seaborn as sns


parser = ArgumentParser()
parser.add_argument('-m', '--metric',
                    choices=["auc", "ks", "pak"],
                    required=True,
                    help="Metric to use")
parser.add_argument('-d', '--dataset',
                    required=True,
                    help="Dataset to train on")
parser.add_argument('-i', '--input',
                    default='eval/viz/trajectories/{dataset}-exactboost_{metric}-{what}-{n_estimators}.npy',
                    help="Path to read data from")
parser.add_argument('-o', '--output',
                    default='eval/viz/trajectories/{dataset}-exactboost_{metric}.pgf',
                    help="Path to save figure to")
parser.add_argument('-S', '--seed',
                    type=int,
                    default=0,
                    help="Random seed to use for UMAP")
parser.add_argument('-s', '--select',
                    type=int,
                    nargs='+',
                    help="Which model to draw the trajectory of")
parser.add_argument('-e', '--n-estimators',
                    type=int,
                    nargs='+')
parser.add_argument('-f', '--fast',
                    action='store_true')
parser.add_argument('-c', '--colormap',
                    default='coolwarm',
                    help="Color palette to use for the plots")
args = parser.parse_args()

np.random.seed(args.seed)


COLORMAP = sns.color_palette(args.colormap, as_cmap=True)

N_ROUNDS = 50


output_path = args.output.format(dataset=args.dataset,
                                 metric=args.metric)

# Plot
def plot_one(n_estimators, selection, xs, ys, metrics, ax):
    hexbins = ax.hexbin(xs, ys, C=1 - metrics, gridsize=25, cmap=COLORMAP)
    hexbins.set_clim(CLIM_MIN, CLIM_MAX)

    ax.set_xticks([])
    ax.set_yticks([])

    if selection is not None:
        js = list(range(N_ROUNDS*selection, N_ROUNDS*(selection+1)))

        unique_pts = np.unique(np.array([xs[js], ys[js]]).T, axis=0)
        xjs, yjs = unique_pts[:, 0], unique_pts[:, 1]
        tck, u = scipy.interpolate.splprep([xjs, yjs], s=5)
        spline_domain = np.linspace(0, 1, 500)
        spline_x, spline_y = scipy.interpolate.splev(spline_domain, tck)
        spline_points = np.array([spline_x, spline_y]).T.reshape(-1, 1, 2)
        spline_segments = np.concatenate([spline_points[:-1], spline_points[1:]], axis=1)

        lc = LineCollection(spline_segments, linewidths=0.75*(1-spline_domain)+0.9, color='black')
        lc.set_capstyle('round')
        ax.add_collection(lc)
        ax.scatter(xjs, yjs, marker='x', lw=0.3, s=30, c='black')

        # for i, j in zip(js[:-1], js[1:]):
        #     ax.arrow(xs[i], ys[i], xs[j]-xs[i], ys[j]-ys[i], linewidth=0.2, head_width=1, length_includes_head=True, fill=True, color='black')

    ax.set_xlim(xs.min()-2, xs.max()+2)
    ax.set_ylim(ys.min()-2, ys.max()+2)


metrics_min = []
metrics_max = []
for train_or_test in ['train', 'test']:
    for n_estimators in args.n_estimators:
        metrics = np.load(args.input.format(dataset=args.dataset,
                                            metric=args.metric,
                                            what='metrics_' + args.metric + "_" + train_or_test,
                                            n_estimators=n_estimators))
        metrics_min.append(metrics.min())
        metrics_max.append(metrics.max())
metrics_min = min(metrics_min)
metrics_max = max(metrics_max)

CLIM_MAX = 1 - metrics_min
CLIM_MIN = 1 - metrics_max

mpl.rcParams['axes.spines.left']   = False
mpl.rcParams['axes.spines.right']  = False
mpl.rcParams['axes.spines.top']    = False
mpl.rcParams['axes.spines.bottom'] = False
mpl.rcParams['font.size']          = 18

fig, axss = plt.subplots(figsize=(4*len(args.n_estimators), 6),
                         nrows=2, ncols=len(args.n_estimators), squeeze=False)
with tqdm(total=2*len(args.n_estimators)) as pg:
    for rowno, (train_or_test, axs) in enumerate(zip(['train', 'test'], axss)):
        for colno, (n_estimators, selection, ax) in enumerate(zip(args.n_estimators, args.select, axs)):
            if rowno == 0:
                ax.set_xlabel(f"$E = {n_estimators}$", fontsize=20)
                ax.xaxis.set_label_position('top')
            if colno == 0:
                ax.set_ylabel(f"{train_or_test.title()}", rotation='vertical', fontsize=18)
            # ax.axis('off')

            xs = np.load(args.input.format(dataset=args.dataset,
                                           metric=args.metric,
                                           what='xs',
                                           n_estimators=n_estimators))
            ys = np.load(args.input.format(dataset=args.dataset,
                                           metric=args.metric,
                                           what='ys',
                                           n_estimators=n_estimators))
            metrics = np.load(args.input.format(dataset=args.dataset,
                                                metric=args.metric,
                                                what='metrics_' + args.metric + "_" + train_or_test,
                                                n_estimators=n_estimators))

            metrics_test = np.load(args.input.format(dataset=args.dataset,
                                                     metric=args.metric,
                                                     what='metrics_' + args.metric + "_test",
                                                     n_estimators=n_estimators))
            final_losses = metrics_test[N_ROUNDS-1::N_ROUNDS]
            max_loss = np.quantile(final_losses, 0.95)
            selected_loss = final_losses[final_losses > max_loss][selection]
            selection, = np.where(final_losses == selected_loss)
            selection = selection[0]

            plot_one(n_estimators, selection, xs, ys, metrics, ax)

            pg.update()

fig.tight_layout()

colorbar = fig.colorbar(ScalarMappable(norm=Normalize(CLIM_MIN, CLIM_MAX), cmap=COLORMAP), ax=axss)
colorbar.set_label(f"{args.metric.upper()} loss", rotation=90)

print("=> save figure")
mpl.use("pgf")
fig.savefig(output_path, bbox_inches="tight")
