import argparse
import copy
import os
import sys

import numpy as np
import pytest
from tqdm import tqdm

import utils as utils
from SVM import FeatureFucntion
from utils import DIVIDER, parse_JSON


def main(args):
    print("building SVM ...")
    svm = FeatureFucntion.load_pickles(args.pickles_dir)

    print("parsing jsons to infer")
    _, programs, _, _ = parse_JSON(args.json_file)

    print("make inference")
    programs = programs[:100]
    val = 0
    length = 0
    for program in tqdm(programs):
        y = svm.inference(program)

        for a, b in zip(program["y_names"], y):
            if a == b:
                val += 1
        length += len(y)

    print("correct percentage -> {:.2%}".format(val * 1.0 / length))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="make inference")
    parser.add_argument("-p", "--pickles", required=True, dest="pickles_dir")
    parser.add_argument("-j", "--json", required=True, dest="json_file")
    args = parser.parse_args()

    main(args)
