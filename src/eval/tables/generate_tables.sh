#!/bin/sh

# Running time benchmark table

python src/eval/running_time_benchmarks/generate.py -d dataset_name -m model_name
python src/eval/running_time_benchmarks/make_table.py -d a1a german gisette gmsc heart ionosphere liver-disorders oil-spill splice svmguide1 -f

# Evaluation (AUC, KS and P@k)

## Main text table: ExactBoost vs exact benchmarks as estimators

python src/eval/tables/eval_models_latex-paper_table.py --only_exact

## Main text table: ensemblers comparison

python src/eval/tables/eval_models_latex-paper_table.py -ens

## Appendix tables (estimators and ensemblers on all datasets)

python src/eval/tables/eval_models_latex.py -m auc --filename table-estimators-auc-full.tex
python src/eval/tables/eval_models_latex.py -m auc -ens --filename table-ensemblers-auc-full.tex
python src/eval/tables/eval_models_latex.py -m ks --filename table-estimators-ks-full.tex
python src/eval/tables/eval_models_latex.py -m ks -ens --filename table-ensemblers-ks-full.tex
python src/eval/tables/eval_models_latex.py -m pak --filename table-estimators-pak-full.tex
python src/eval/tables/eval_models_latex.py -m pak -ens --filename table-ensemblers-pak-full.tex

# Dataset characteristics tables
## Both full table presented in the appendix and table with only the main text datasets.

python src/eval/tables/datasets_latex.py
