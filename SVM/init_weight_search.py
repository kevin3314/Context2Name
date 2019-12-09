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
    STEP_PARA = 0.1
    for proportion in np.arange(0.1, 0.7, 0.1):
        print("proportion is {}".format(proportion))
        val = 0
        length = 0
        for i, v in enumerate(kf.split(json_files)):
            if args.s and i > 0:
                break
            step_seq = utils.sqrt_sequence(STEP_PARA)
            print(f"STEP_PAR is {STEP_PARA}")
            train = v[0]
            test = v[1]
            print("start {} fold".format(i))
            train_datas = list(json_files[train])
            function_keys, programs, candidates, label_seq_dict = parse_JSON(train_datas)

            svm = FeatureFucntion(function_keys, candidates, label_seq_dict)

            svm.subgrad(
                programs,
                step_seq,
                utils.naive_loss,
                iterations=30,
                BETA=0.5,
                init_weight_proportion=proportion,
            )

            test_datas = list(json_files[test])
            _, test_programs, _, _ = parse_JSON(test_datas)
            with Pool() as pool:
                res = pool.map(svm.inference_only_correct_number, test_programs)

            tmp_val, tmp_length = (sum(x) for x in zip(*res))
            if i == 0:
                print("svm.weight -> {}".format(svm.weight[:40]))
            val += tmp_val
            length += tmp_length

        correct_per = val * 1.0 / length
        print("score -> {}".format(correct_per))
        para_map[str(proportion)] = correct_per
    print("The Best  proportion -> {}".format(max(para_map, key=para_map.get)))
    print(para_map)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train to get weight")
    parser.add_argument("-j", "--json", required=True, dest="json_files")
    parser.add_argument("-s", action="store_true")
    # parser.add_argument("-p", "--pickles", required=False, dest="pickles_dir")
    args = parser.parse_args()

    main(args)
