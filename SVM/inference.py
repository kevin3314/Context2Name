import argparse
import copy
import os
import sys

import numpy as np
import pytest

import utils as utils
from SVM import FeatureFucntion
from utils import DIVIDER, parse_JSON


def main(args):
    # parse json files
    print("parsing JSON files ...")
    function_keys, programs, candidates, label_seq_dict = parse_JSON(args.input_dir)

    print("building SVM ...")
    svm = FeatureFucntion(function_keys, candidates, label_seq_dict, weight_path=args.pre_weight)

    print("make inference")
    _, programs, _, _ = parse_JSON(args.json_file)
    program = programs[0]

    y = svm.inference(program)

    print("----------- correct --------------")
    print(program["y_names"])

    print("----------- inference --------------")
    print(y)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="make inference")
    parser.add_argument("-i", "--input", required=True, dest="input_dir")
    parser.add_argument("-j", "--json", required=True, dest="json_file")
    parser.add_argument("-w", "--weight", required=False, dest="pre_weight")
    args = parser.parse_args()

    main(args)
