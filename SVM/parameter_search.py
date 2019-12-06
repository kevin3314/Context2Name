import argparse
import os
from multiprocessing import Pool

import numpy as np
from sklearn.model_selection import KFold
from tqdm import tqdm

import utils
from SVM import FeatureFucntion
from utils import parse_JSON


def main(args):
    json_files = [
        os.path.join(args.json_files, x)
        for x in os.listdir(args.json_files)
        if not x.startswith(".") and x[-5:] == ".json"
    ]
    json_files = np.array(json_files)
    kf = KFold(n_splits=10)

    # experiment for parameter.
    para_map = {}
    for GUNMA in np.arange(0.3, 0.8, 0.1):
        print("GUNMA is {}".format(GUNMA))
        val = 0
        length = 0
        for i, v in enumerate(kf.split(json_files)):
            step_seq = utils.simple_sequence(GUNMA)
            train = v[0]
            test = v[1]
            print("start {} fold".format(i))
            train_datas = list(json_files[train])
            function_keys, programs, candidates, label_seq_dict = parse_JSON(train_datas)

            svm = FeatureFucntion(function_keys, candidates, label_seq_dict)

            svm.subgrad(
                programs,
                step_seq,
                # utils.simple_sequence(0.03),
                utils.naive_loss,
                iterations=30,
                BETA=0.5,
            )

            test_datas = list(json_files[test])
            _, test_programs, _, _ = parse_JSON(test_datas)
            with Pool() as pool:
                res = pool.map(svm.inference_only_correct_number, test_programs)

            tmp_val, tmp_length = (sum(x) for x in zip(*res))
            if i == 0:
                print("svm.weight -> {}".format(svm.weight[:40]))
            print("tmp_val -> {}".format(tmp_val))
            val += tmp_val
            length += tmp_length

        correct_per = val * 1.0 / length
        print(correct_per)
        para_map[str(GUNMA)] = correct_per
    print("The Best  GUNMA -> {}".format(max(para_map, key=para_map.get)))
    print(para_map)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train to get weight")
    parser.add_argument("-j", "--json", required=True, dest="json_files")
    # parser.add_argument("-p", "--pickles", required=False, dest="pickles_dir")
    args = parser.parse_args()

    main(args)
