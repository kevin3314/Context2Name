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
    function_keys, programs, candidates, label_seq_dict = parse_JSON(args.json_files)

    print("building SVM ...")
    svm = FeatureFucntion(function_keys, candidates, label_seq_dict)

    print("start lerning!")
    svm.subgrad(
        programs,
        utils.simple_sequence(0.03),
        utils.naive_loss,
        iterations=30,
        save_dir=args.output_dir,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train to get weight")
    parser.add_argument("-j", "--json", required=True, dest="json_files")
    parser.add_argument("-o", "--output", required=True, dest="output_dir")
    # parser.add_argument("-p", "--pickles", required=False, dest="pickles_dir")
    args = parser.parse_args()

    main(args)
