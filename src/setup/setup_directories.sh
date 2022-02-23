#!/bin/bash

current_dir=${PWD##*/}

if [[ $current_dir != "exactboost" ]]; then
    echo "All scripts must be run from base directory 'exactboost'"
    exit
fi

declare -a subdirs=("data" "eval" "models" "setup")
for subdir in "${subdirs[@]}"
do
    # Make directory creating parent directories if necessary
    cmd="mkdir -p $subdir"
    echo "$cmd"; $cmd
done

