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
    # parse json files
    print("parsing JSON files ...")
    function_keys, programs, candidates, label_seq_dict = parse_JSON(args.input_dir)

    print("building SVM ...")
    svm = FeatureFucntion(function_keys, candidates, label_seq_dict, weight_path=args.pre_weight)

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
    parser.add_argument("-i", "--input", required=True, dest="input_dir")
    parser.add_argument("-j", "--json", required=True, dest="json_file")
    parser.add_argument("-w", "--weight", required=False, dest="pre_weight")
    args = parser.parse_args()

    main(args)
