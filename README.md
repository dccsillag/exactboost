# ExactBoost

This repository contains the code to reproduce all figures and tables in "ExactBoost: Directly Boosting the Margin in Combinatorial and Non-decomposable Metrics," published at AISTATS 2022.

```bibtex
@inproceedings{csillag2022exactboost,
  title={ExactBoost: Directly Boosting the Margin in Combinatorial and Non-decomposable Metrics},
  author={Daniel Csillag and Carolina Piazza and Thiago Ramos and João Vitor Romano and Roberto Oliveira and Paulo Orenstein},
  booktitle={AISTATS},
  year={2022}
}
```

## Project Setup Instructions

- All scripts should be run from the Git repository root directory (`exactboost/`).
- Run the script `src/setup/setup_directories.sh` to create the necessary directories for the project. See "Folder Contents" below.
- Run `conda env create -f src/setup/env_exactboost.yml` to create a new conda environment with all necessary packages.
- Run `conda activate env_exactboost` to activate the environment.

### Compiling the C++ Code

The code in `src/models/exactboost/` is written in C++17, so a conforming compiler is required. Other than that, the dependencies are all managed by the Conda environment.

The code can be compiled with [make](https://pubs.opengroup.org/onlinepubs/9699919799/utilities/make.html), and generates a dynamic library that can be loaded with Python as a module.

```
make -C src/models/exactboost/
```

### Making TopPush available through python

The [implementation of TopPush](https://web.archive.org/web/20210123175252/https://github.com/VaclavMacha/ClassificationOnTop.jl) used as a benchmark for the precision at k loss is written in julia. In order to access it through python, refer to the following steps:
- Add `export PATH="$PATH:$(readlink -f {path_to_repo}/setup/julia-1.5.3/bin)` to your shell rc, where `{path_to_repo}` is the path to the cloned repository (exactboost).
- Reload your shell rc via `.` or by logging out and back in, for example.
- Run `src/setup/setup_top_push.sh`. It should take a while to complete. For different setups, this shell script can be used as a guide.

## Recreating Figures and Tables

Note the figures are generated as [PGF](https://pgf-tikz.github.io/) to be easily integrated to the Latex document. They can be changed to [PNG](https://www.w3.org/TR/PNG/) by changing the extension in Matplotlib's `plt.savefig("filename.pgf")` to `plt.savefig("filename.png")`.

### Downloading the datasets

- To download and process the needed datasets, run `src/data/paper.sh`.
- Manually download the `gmsc` and `cskaggle` datasets (which require a Kaggle login) and the `mq2008` dataset (for which no direct link is provided). The download pages can be found in `src/data/download.py`. The files needed are `cs-training.csv` for `gmsc`, `application_train.csv` for `cskaggle` and `min.txt` for `mq2008`. Once all files are placed in their corresponding `data/raw/{dataset}/` directories, process the datasets using `src/data/process.py -d gmsc`, `src/data/process.py -d cskaggle` and `src/data/process.py -d mq2008`.

### Figure 1

- Create the required margin data by running `src/eval/margin_plots/generate.py` with flags `-d` indicating the dataset and `-m` indicating the metric (`auc`, `ks` or `pak`). Hyperparameters can be chosen through flags as well, but they are set to values used in the paper by default.
- Figure 2 is then created by running `src/eval/margin_plots/paper_plot.py`. The default datasets and metrics are hardcoded: `svmguide1` for `auc`, `gmsc` for `ks` and `splice` for `pak`.

### Figure 2

- Create the trajectory plot data by running `src/eval/trajectory_plots/generate.py -m ks -d heart`.
- Get the evaluation data for each point via `src/eval/trajectory_plots/evaluate.py -m ks -d heart -e $N_RUNS`. The flag `-e` contains the number of runs to average over. In the paper, it takes the values 1, 2, 10, 100, 250.
- Project the data into 2D via `src/eval/trajectory_plots/project.py -m ks -d heart -e $N_RUNS`, where `metric` is either `auc`, `ks` or `pak`.
- Finally, the plot is obtained through `src/eval/trajectory_plots/plot.py -m ks -d heart -s $SELECTED_TRAJECTORIES -e $N_RUNS -c $COLOR_SCHEME`, where `$SELECTED_MODELS` indicate for which model numbers to draw the trajectory for and `$COLOR_SCHEME` give the `seaborn` color pallete for the plot. In the paper, we use: `python src/eval/trajectory_plots/plot.py -m ks -d heart -s 7 20 16 14 3 -e 1 2 10 100 250 -c Spectral`.

### Figure 3

- Create the cross-validation data by running `src/eval/boxplots/do_boxplots.sh`. The script takes the flags `-d` to indicate which datasets to use, `-m` to define the estimators and `-M` to define ensemblers. Estimators whose predictions are used for ensembling are hardcoded in the script and running `do_boxplots.sh` for them is a requirement before ensembling.
- Figure 3 can then be created by running `src/eval/benchmark_plots/plot.py --benchmarks_errorbar --free_scale`.

### Table 1, Table 2, Table 3 and Supplementary Material tables

- Results presented in the evaluation tables come from 5-fold cross-validations generated by the boxplots pipeline. Therefore, `src/eval/boxplots/do_boxplots.sh` must be run for all models and datasets before proceeding.
- All the tables for the main paper and the supplementary material can then be generated via `src/eval/tables/generate_tables.sh`.

## Folder Contents

- `data`: raw and processed datasets.
- `eval`: model performance metrics, plots, tables and other output used to evaluate models.
- `models`: trained models for future predictions and files created by models, including intermediate steps or for testing purposes (e.g., selection among different model experiments).
- `setup`: customised Julia binary necessary to run TopPush (not needed for ExactBoost nor for the other models in the project).
- `src`: project source code (pushed to the repository).
