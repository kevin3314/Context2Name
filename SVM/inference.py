import argparse
import copy
import os
import sys
from multiprocessing import Pool

import numpy as np
import pytest
from tqdm import tqdm

import utils as utils
from SVM import FeatureFucntion
from utils import DIVIDER, parse_JSON


def main(args):
    print("building SVM ...")
    svm = FeatureFucntion.load_pickles(args.pickles_dir)

    print(svm.weight[:1])

    print("parsing jsons to infer")
    _, programs, _, _ = parse_JSON(args.json_file)

    print("make inference")
    with Pool() as pool:
        res = list(tqdm(pool.imap_unordered(svm.inference_only_correct_number, programs), total=len(programs)))
    val, length = (sum(x) for x in zip(*res))

    print("correct percentage -> {:.2%}".format(val * 1.0 / length))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="make inference")
    parser.add_argument("-p", "--pickles", required=True, dest="pickles_dir")
    parser.add_argument("-j", "--json", required=True, dest="json_file")
    args = parser.parse_args()

    main(args)
