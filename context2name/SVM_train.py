# import os.path
# import operator
import argparse
import collections
import json
import pickle

import numpy as np


class FeatureFucntion:
    """Class for feature function.

    Attributes:
        function_keys(list):
            function_keys is list of feature.
            feature like: (x, y, ["ArrayExpression", "CallExpression"])

        weight(np.ndarray):
            weight is weight to be learned.
    """
    def __init__(self, function_keys):
        self.function_keys = function_keys
        self.weight = np.ones(len(function_keys))

    def eval(self, key):
        if key in self.function_keys:
            index = self.function_keys.index(key)
            return self.weight[index]


def main(args):
    with open(args.json, "r") as f:
        jsonData = json.load(f)
    function_keys = []
    programs = []
    for key in jsonData:
        program = jsonData[key]
        programs.append(program)

        for key2 in program:
            if key2 == "y_names":
                continue
            obj = program[key2]
            k = set([obj["xName"], obj["yName"]])
            seq = obj["sequence"]
            function_keys.append((k, seq))

    func = FeatureFucntion(function_keys)
    test_key = set(["end", "t"])
    test_ary = ["MemberExpression"]
    print(func.eval((test_key, test_ary)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train to get weight")
    parser.add_argument("-j", "--json", required=True, dest="json")
    # parser.add_argument("-o", "--output", required=True, dest="output")
    args = parser.parse_args()

    main(args)
