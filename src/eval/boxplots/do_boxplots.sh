#!/bin/bash

N_JOBS=1
N_JOBS_MODEL=-1
MARGIN=0.05

while getopts "d:m:M:vrGVCEPj:J:t:" name
do
    case $name in
        d) DATASETS="$DATASETS $OPTARG" ;;
        v) VAL=-val ;;
        r) RIE=-rie ;;
        G) NOGENERATE=1 ;;
        V) NOEVALUATE=1 ;;
        C) NOCOMBINE=1 ;;
        E) NOENSEMBLE=1 ;;
        j) N_JOBS_MODEL=$OPTARG ;;
        J) N_JOBS=$OPTARG ;;
        m) MODELS="$MODELS $OPTARG" ;;
        M) ENSEMBLE_MODELS="$ENSEMBLE_MODELS $OPTARG" ;;
        t) MARGIN=$OPTARG ;;
        ?)
            echo "No such command line option -- available options are:"
            echo "  -d          pass dataset to Python scripts"
            echo "  -v          pass -val to the Python scripts"
            echo "  -r          pass -rie to the Python scripts"
            echo "  -G          do not run generate"
            echo "  -V          do not run evaluate"
            echo "  -C          do not run combine"
            echo "  -E          do not run ensemble"
            echo "  -J          number of jobs for parallelizing on splits"
            echo "  -j          number of jobs for model parallelization"
            echo "  -m          add a model as an estimator"
            echo "  -M          add a model as an ensembler"
            echo "  -t          value of margin to use"
            exit 2
            ;;
    esac
done

set -e
set -x

COMBINE_MODELS='adaboost knn logistic neural_network random_forest xgboost'

echo $(date)

for dataset in $DATASETS
do
    for model in $MODELS
    do
        # Generate (test)
        [ -z "$NOGENERATE" ] && python src/eval/boxplots/generate.py -d "$dataset" -m "$model" -j $N_JOBS -jm $N_JOBS_MODEL -ts test $RIE $VAL -mt $MARGIN

        # Generate (ens)
        [ -z "$NOGENERATE" ] && [[ "$model" != *"top_push"* ]] && [[ "$model" != *"rankboost"* ]] && [[ "$model" != *"dmks"* ]] && [[ "$model" != *"exactboost"* ]] && python src/eval/boxplots/generate.py -d "$dataset" -m "$model" -j $N_JOBS -jm $N_JOBS_MODEL -ts ens $RIE $VAL -mt $MARGIN

        # Evaluate (test)
        [ -z "$NOEVALUATE" ] && python src/eval/boxplots/evaluate.py -d "$dataset" -m "$model" $VAL
    done

    # Combine (test)
    [ -z "$NOCOMBINE" ] && python src/eval/boxplots/combine.py -d "$dataset" -m $COMBINE_MODELS -j $N_JOBS -ts test $VAL

    # Combine (ens)
    [ -z "$NOCOMBINE" ] && python src/eval/boxplots/combine.py -d "$dataset" -m $COMBINE_MODELS -j $N_JOBS -ts ens $VAL

    # Ensemble
    for ensemble_model in $ENSEMBLE_MODELS
    do
        # Generate
        [ -z "$NOENSEMBLE" ] && python src/eval/boxplots/ensemble.py -d "$dataset" -e "$ensemble_model" -j $N_JOBS -jm $N_JOBS_MODEL $RIE $VAL -mt $MARGIN

        # Evaluate
        [ -z "$NOEVALUATE" ] && python src/eval/boxplots/evaluate.py -m "$ensemble_model"_ensembler -d "$dataset" $VAL
    done
done

echo $(date)
