#!/bin/sh

# This is a helper script to download.py and process.py most datasets used in the paper,
# Exceptions are `cskaggle` and `gmsc`, which require manual download from kaggle's website,
# and `mq2008`, which requires manual download from project LETOR's page.

# Download

python src/data/download.py -d a1a
python src/data/download.py -d australian
python src/data/download.py -d banknote
python src/data/download.py -d breast-cancer
python src/data/download.py -d cod-rna
python src/data/download.py -d colon-cancer
python src/data/download.py -d covtype
#python src/data/download.py -d cskaggle
python src/data/download.py -d diabetes
python src/data/download.py -d fourclass
python src/data/download.py -d german
python src/data/download.py -d gisette
#python src/data/download.py -d gmsc
python src/data/download.py -d heart
python src/data/download.py -d housing
python src/data/download.py -d ijcnn1
python src/data/download.py -d ionosphere
python src/data/download.py -d liver-disorders
python src/data/download.py -d madelon
python src/data/download.py -d mammography
#python src/data/download.py -d mq2008
python src/data/download.py -d oil-spill
python src/data/download.py -d phishing
python src/data/download.py -d phoneme
python src/data/download.py -d skin-nonskin
python src/data/download.py -d sonar
python src/data/download.py -d splice
python src/data/download.py -d svmguide1
python src/data/download.py -d svmguide3
python src/data/download.py -d taiwan
python src/data/download.py -d w1a

# Process

python src/data/process.py -d a1a
python src/data/process.py -d australian
python src/data/process.py -d banknote
python src/data/process.py -d breast-cancer
python src/data/process.py -d cod-rna
python src/data/process.py -d colon-cancer
python src/data/process.py -d covtype
#python src/data/process.py -d cskaggle
python src/data/process.py -d diabetes
python src/data/process.py -d fourclass
python src/data/process.py -d german
python src/data/process.py -d gisette
#python src/data/process.py -d gmsc
python src/data/process.py -d heart
python src/data/process.py -d housing
python src/data/process.py -d ijcnn1
python src/data/process.py -d ionosphere
python src/data/process.py -d liver-disorders
python src/data/process.py -d madelon
python src/data/process.py -d mammography
#python src/data/process.py -d mq2008
python src/data/process.py -d oil-spill
python src/data/process.py -d phishing
python src/data/process.py -d phoneme
python src/data/process.py -d skin-nonskin
python src/data/process.py -d sonar
python src/data/process.py -d splice
python src/data/process.py -d svmguide1
python src/data/process.py -d svmguide3
python src/data/process.py -d taiwan
python src/data/process.py -d w1a

