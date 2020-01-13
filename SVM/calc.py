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

    t_list = []
    for program in programs:
        for key, obj in program.items():
            if key == "y_names":
                continue

            t = ()
            if obj["type"] == "var-var":
                x = obj["xName"]
                y = obj["yName"]
                xscope = obj["xScopeId"]
                yscope = obj["yScopeId"]
                seq = obj["sequence"]
                t = (x, y, xscope, yscope, seq)
            else:
                x = obj["xName"]
                xscope = obj["xScopeId"]
                y = obj["yName"]
                seq = obj["sequence"]
                t = (x, xscope, y, seq)
            t_list.append(t)

    c = Counter(t_list)
    c_values = list(c.values())
    print(c_values[:30])
    c_values.sort(reverse=True)
    print(c_values[:30])
    print(max(c, key=c.get))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train to get weight")
    parser.add_argument("-j", "--json", required=True, dest="json_files")
    args = parser.parse_args()

    main(args)
