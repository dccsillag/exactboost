import argparse

from src.utils.general_utils import get_folder
from src.utils.data_utils import download_from_url, simple_download_from_url


URL = {
    "a1a": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary",
    "a2a": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary",
    "a3a": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary",
    "a4a": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary",
    "a5a": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary",
    "a6a": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary",
    "a7a": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary",
    "a8a": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary",
    "a9a": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary",
    "australian":
        "http://archive.ics.uci.edu/ml/machine-learning-databases/statlog/australian/",
    "avazu": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary",
    "banknote": "https://archive.ics.uci.edu/ml/machine-learning-databases/00267",
    "breast-cancer": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary",
    "cod-rna": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary",
    "colon-cancer": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary",
    "covtype": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary",
    "criteo": "https://s3-us-west-2.amazonaws.com/criteo-public-svm-data",
    "cskaggle": "https://www.kaggle.com/c/home-credit-default-risk/data",
    "diabetes": "https://raw.githubusercontent.com/jbrownlee/Datasets/master",
    "duke-breast-cancer":
        "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary",
    "epsilon": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary",
    "fourclass": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary",
    "german": "https://online.stat.psu.edu/onlinecourses/sites/stat508/files",
    "gisette": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary",
    "gmsc": "https://www.kaggle.com/c/GiveMeSomeCredit/data",
    "heart": "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease",
    "higgs": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary",
    "housing": "https://archive.ics.uci.edu/ml/machine-learning-databases/housing",
    "ijcnn1": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary",
    "ionosphere": "https://raw.githubusercontent.com/jbrownlee/Datasets/master",
    "kdd": "https://www.kdd.org/kdd-cup/view/kdd-cup-2008/Data",
    "liver-disorders": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary",
    "madelon": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary",
    "mammography": "https://raw.githubusercontent.com/jbrownlee/Datasets/master",
    "mushrooms": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary",
    "mq2008": ("https://www.microsoft.com/en-us/research/project/"
               "letor-learning-rank-information-retrieval/"),
    "news20": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary",
    "oil-spill": "https://raw.githubusercontent.com/jbrownlee/Datasets/master",
    "phishing": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary",
    "phoneme": "https://raw.githubusercontent.com/jbrownlee/Datasets/master",
    "rcv": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary",
    "real-sim": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary",
    "skin-nonskin": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary",
    "spambase": "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase",
    "splice": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary",
    "sonar": "https://raw.githubusercontent.com/jbrownlee/Datasets/master",
    "susy": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary",
    "svmguide1": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary",
    "svmguide3": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary",
    "taiwan": "https://archive.ics.uci.edu/ml/machine-learning-databases/00350",
    "url": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/",
    "w1a": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/",
    "w2a": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/",
    "w3a": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/",
    "w4a": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/",
    "w5a": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/",
    "w6a": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/",
    "w7a": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/",
    "w8a": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/",
    "w9a": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/",
    "webspam": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary",
}

FILES = {
    "a1a": ["a1a"],
    "a2a": ["a2a"],
    "a3a": ["a3a"],
    "a4a": ["a4a"],
    "a5a": ["a5a"],
    "a6a": ["a6a"],
    "a7a": ["a7a"],
    "a8a": ["a8a"],
    "a9a": ["a9a"],
    "australian": ["australian.dat"],
    "avazu": ["avazu-app.bz2"],
    "banknote": ["data_banknote_authentication.txt"],
    "breast-cancer": ["breast-cancer"],
    "cod-rna": ["cod-rna"],
    "colon-cancer": ["colon-cancer.bz2"],
    "covtype": ["covtype.libsvm.binary.bz2"],
    "criteo": ["criteo.kaggle2014.svm.tar.gz"],
    "cskaggle": [""],
    "diabetes": ["pima-indians-diabetes.data.csv"],
    "duke-breast-cancer": ["duke.bz2"],
    "epsilon": ["epsilon_normalized.bz2"],
    "fourclass": ["fourclass"],
    "german": ["german_credit.csv"],
    "gisette": ["gisette_scale.bz2"],
    "gmsc": [""],
    "heart": ["processed.cleveland.data"],
    "higgs": ["HIGGS.bz2"],
    "housing": ["housing.data"],
    "ijcnn1": ["ijcnn1.bz2"],
    "ionosphere": ["ionosphere.csv"],
    "kdd": [""],
    "liver-disorders": ["liver-disorders"],
    "madelon": ["madelon"],
    "mammography": ["mammography.csv"],
    "mushrooms": ["mushrooms"],
    "mq2008": [""],
    "news20": ["news20.binary.bz2"],
    "oil-spill": ["oil-spill.csv"],
    "phishing": ["phishing"],
    "phoneme": ["phoneme.csv"],
    "rcv": ["rcv1_train.binary.bz2"],
    "real-sim": ["real-sim.bz2"],
    "skin-nonskin": ["skin_nonskin"],
    "spambase": ["spambase.data", "spambase.names"],
    "splice": ["splice"],
    "sonar": ["sonar.csv"],
    "susy": ["SUSY.bz2"],
    "svmguide1": ["svmguide1"],
    "svmguide3": ["svmguide3"],
    "taiwan": ["default%20of%20credit%20card%20clients.xls"],
    "url": ["url_combined.bz2"],
    "w1a": ["w1a"],
    "w2a": ["w2a"],
    "w3a": ["w3a"],
    "w4a": ["w4a"],
    "w5a": ["w5a"],
    "w6a": ["w6a"],
    "w7a": ["w7a"],
    "w8a": ["w8a"],
    "w9a": ["w9a"],
    "webspam": ["webspam_wc_normalized_trigram.svm.bz2"],
}


parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset", "-d", default=None, help="Which dataset to download.",
)
args = parser.parse_args()


DATASET = args.dataset
output_folder = get_folder(f"data/raw/{DATASET}")

for file in FILES[DATASET]:
    if file == "":
        raise ValueError("Cannot retrieve the data programatically. "
                         f"Please download it from {URL[DATASET]}.")
        continue

    file_url = f"{URL[DATASET]}/{file}"
    file_path = f"{output_folder}/{file}"

    try:
        download_from_url(file_url, file_path)
    except KeyError:
        simple_download_from_url(file_url, file_path)
