import argparse
import os

import numpy as np
from sklearn.model_selection import KFold
from tqdm import tqdm

import utils
from SVM import FeatureFucntion
from utils import parse_JSON


def main(args):
    json_files = [os.path.join(args.json_files, x) for x in os.listdir(args.json_files)]
    json_files = np.array(json_files)
    kf = KFold(n_splits=10)
    val = 0
    length = 0
    for train, test in kf.split(json_files):
        train_datas = list(json_files[train])
        function_keys, programs, candidates, label_seq_dict = parse_JSON(train_datas)

        print("building SVM ...")
        svm = FeatureFucntion(function_keys, candidates, label_seq_dict)

        print("start lerning!")
        svm.subgrad(
            programs,
            utils.simple_sequence(0.03),
            utils.naive_loss,
            iterations=30,
            LAMBDA=0.5,
            BETA=0.5,
        )

        test_datas = list(json_files[test])
        _, test_programs, _ = parse_JSON(test_datas)
        for program in programs:
            y = svm.inference(program)

            for a, b in zip(program["y_names"], y):
                if a == b:
                    val += 1
            length += len(y)
    print(val * 1.0 / length)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train to get weight")
    parser.add_argument("-j", "--json", required=True, dest="json_files")
    # parser.add_argument("-p", "--pickles", required=False, dest="pickles_dir")
    args = parser.parse_args()

    main(args)
