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

    impossibles = 0

    for program in programs:
        y_names = set(program["y_names"])
        for key, obj in program.items():
            if key == "y_names":
                continue

            keys = []
            if obj["type"] == "var-var":
                y_seq = str(obj["yScopeId"]) + DIVIDER + obj["yName"]
                keys.append(y_seq)
            x_seq = str(obj["xScopeId"]) + DIVIDER + obj["xName"]
            keys.append(x_seq)
            for v in keys:
                if v in y_names:
                    y_names.remove(v)
        impossibles += len(y_names)
    print(impossibles)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train to get weight")
    parser.add_argument("-j", "--json", required=True, dest="json_files")
    args = parser.parse_args()

    main(args)
