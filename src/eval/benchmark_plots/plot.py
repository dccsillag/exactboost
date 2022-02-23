import argparse
import os
import subprocess as sp

import numpy as np
import pandas as pd
import plotnine as pn

from src.utils.general_utils import get_folder

parser = argparse.ArgumentParser()
parser.add_argument(
    "--models",
    "-m",
    nargs="+",
    default=[
        "adaboost",
        "knn",
        "logistic",
        "neural_network",
        "random_forest",
        "xgboost",
    ]
)
parser.add_argument("--models_auc", "-ks", nargs="+", default=[])
parser.add_argument("--models_ks", "-auc", nargs="+", default=[])
parser.add_argument("--models_pak", "-pak", nargs="+", default=[])
parser.add_argument("--exactboost_errorbar", action="store_true")
parser.add_argument("--benchmarks_errorbar", action="store_true")
parser.add_argument("--free_scale", action="store_true")
parser.add_argument("--no_axis_elements", action="store_true")
parser.add_argument(
    "--datasets",
    "-d",
    nargs="+",
    default=[
        "a1a",
        "german",
        "gisette",
        "gmsc",
        "heart",
        "ionosphere",
        "liver-disorders",
        "oil-spill",
        "splice",
        "svmguide1",
    ],
)
args = parser.parse_args()

def col_func(s):
    if s == "auc":
        return "AUC"
    elif s == "ks":
        return "KS"
    else:
        return "P@k"

MODELS = {
    "auc": args.models_auc + args.models,
    "ks": args.models_ks + args.models,
    "pak": args.models_pak + args.models,
}
DATASETS = args.datasets

df = pd.DataFrame()
df_exactboost = pd.DataFrame()

for dataset in DATASETS:
    for metric in ["auc", "ks", "pak"]:
        # Benchmark models
        for model in MODELS[metric]:
            path = f"eval/boxplots-cv/{dataset}/{model}-cv-test-5-{metric}.csv"
            if os.path.exists(path):
                data = np.loadtxt(path)
                data = 1 - data
                df = df.append([[metric, dataset, model, data.mean(), data.std()]])
            else:
                print(f" - Warning: file {path} not found.")
        # ExactBoost models
        path = f"eval/boxplots-cv/{dataset}/exactboost_{metric}-cv-test-5-{metric}.csv"
        if os.path.exists(path):
            data = np.loadtxt(path)
            data = 1 - data
            df_exactboost = df_exactboost.append(
                [[metric, dataset, data.mean(), data.std()]]
            )
        else:
            print(f" - Warning: file {path} not found.")

df.columns = ["metric", "dataset", "model", "benchmark_avg", "benchmark_std"]
df_exactboost.columns = ["metric", "dataset", "exactboost", "exactboost_std"]

df["y_min"] = df["benchmark_avg"] - df["benchmark_std"]
df["y_max"] = df["benchmark_avg"] + df["benchmark_std"]
df = df.drop("benchmark_std", axis=1)

df_exactboost["x_min"] = df_exactboost["exactboost"] - df_exactboost["exactboost_std"]
df_exactboost["x_max"] = df_exactboost["exactboost"] + df_exactboost["exactboost_std"]

df = df.merge(df_exactboost)

# Define offset factor to alleviate collisions in plot.
# No point will be further than half its standard deviation.
n_models = df["model"].nunique()
offset_factor = np.linspace(-0.5, 0.5, n_models)

for i, model in enumerate(df["model"].unique()):
    df.loc[df["model"] == model, "exactboost"] += \
        df.loc[df["model"] == model, "exactboost_std"] * offset_factor[i]

df = df.replace(
    {
        "exactboost_auc": "ExactBoost",
        "exactboost_ks": "ExactBoost",
        "exactboost_pak": "ExactBoost",
        "rankboost": "RankBoost",
        "dmks": "DMKS",
        "top_push": "TopPush",
        "adaboost": "AdaBoost",
        "knn": "kNN",
        "logistic": "Logistic",
        "neural_network": "Neural Net",
        "random_forest": "Random Forest",
        "xgboost": "XGBoost",
    }
)

# Represent models as categorical variables.
# This ensures that legend items are displayed in the desired order.
df = df.assign(
    model=pd.Categorical(
        df["model"],
        sorted(df["model"].unique(), key=str.casefold)  # case-insensitive sorting
    )
)

# Sort models in dataset to guarantee consistent behaviour where points intersect.
df = df.sort_values("model", ascending=False)

plot = (
    pn.ggplot(df, pn.aes("exactboost", "benchmark_avg", color="model"))
    + pn.geom_abline(intercept=0, slope=1, alpha=0.6)
    + (
        pn.geom_errorbar(
            pn.aes(x="exactboost", ymin="y_min", ymax="y_max", width=0.01),
            color="black",
            alpha=0.2,
        )
        if args.benchmarks_errorbar
        else pn.geom_blank()
    )
    + (
        pn.geom_errorbarh(
            pn.aes(y="benchmark_avg", xmin="x_min", xmax="x_max", height=0.05),
            color="black",
            alpha=0.2,
        )
        if args.exactboost_errorbar
        else pn.geom_blank()
    )
    + pn.geom_point(size=2.5)
    + (
        pn.facet_wrap("~metric", labeller=pn.labeller(cols=col_func), scales="free")
        if args.free_scale
        else pn.facet_grid("~metric", labeller=pn.labeller(cols=col_func))
    )
    + (
        pn.scale_color_manual(values=[
            "#1eae98", "#50388d", "#99154e", "#eb4034", "#fd8c04", "#fecd1a"]
        )
        if df["model"].nunique() <= 6
        else pn.geom_blank()
    )
    + pn.theme_bw()
    + pn.ylab("Benchmark loss")
    + pn.xlab("ExactBoost loss")
    + (
        pn.scale_y_continuous(breaks=np.linspace(
            df["benchmark_avg"].min().round(1),
            df["benchmark_avg"].max().round(1),
            4)
        )
        if args.free_scale and args.benchmarks_errorbar
        else pn.geom_blank()
    )
    + pn.theme(
        text=pn.element_text(size=16),
        legend_key=pn.element_blank(),
        legend_background=pn.element_blank(),
        legend_box_background=pn.element_rect(fill="#fffafa"),
        legend_title=pn.element_blank(),
        legend_text=pn.element_text(size=16),
        panel_spacing_x=0.5,
        axis_text_x=(pn.element_blank() if args.no_axis_elements else pn.element_text()),
        axis_text_y=(pn.element_blank() if args.no_axis_elements else pn.element_text()),
        axis_ticks_major_y=(pn.element_blank() if args.no_axis_elements else pn.element_line()),
        axis_ticks_major_x=(pn.element_blank() if args.no_axis_elements else pn.element_line()),
        axis_ticks_minor_y=(pn.element_blank() if args.no_axis_elements else pn.element_line()),
        axis_ticks_minor_x=(pn.element_blank() if args.no_axis_elements else pn.element_line()),
    )
)

path = get_folder("eval/benchmark_plots")

plot.save(f"{path}/benchmark_plots.pgf", width=17, height=4, verbose=False, bbox_inches="tight")

# Remove font definitions from generated pgf file. This ensures the figure
# inherits the parent LaTeX document font.
sp.call(["sed", "-i", r"s/\\setmainfont{[^\}]*}\\..family//g", f"{path}/benchmark_plots.pgf"])
