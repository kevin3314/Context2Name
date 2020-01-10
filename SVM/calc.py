import argparse
import copy
import os
import sys
from collections import Counter

import numpy as np
import pytest

import utils as utils
from SVM import FeatureFucntion
from utils import DIVIDER, parse_JSON, Triplet


def main(args):
    # parse json files
    print("parsing JSON files ...")
    function_keys, programs, candidates, label_seq_dict = parse_JSON(args.json_files)

    triplet_list = []
    for program in programs:
        for key, obj in program.items():
            if key == "y_names":
                continue
            x = obj["xName"]
            y = obj["yName"]
            seq = obj["sequence"]
            key_name = Triplet(x, seq, y)
            triplet_list.append(key_name)

    c = Counter(triplet_list)
    c_values = list(c.values())
    print(c_values[:30])
    c_values.sort(reverse=True)
    print(c_values[:30])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train to get weight")
    parser.add_argument("-j", "--json", required=True, dest="json_files")
    args = parser.parse_args()

    main(args)
