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

    sum_edges = sum((len(program)-1) for program in programs)
    print(sum_edges / len(programs))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train to get weight")
    parser.add_argument("-j", "--json", required=True, dest="json_files")
    args = parser.parse_args()

    main(args)
