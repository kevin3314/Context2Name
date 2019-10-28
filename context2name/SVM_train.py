# import os.path
# import operator
import argparse
import collections
import json
import pickle
import copy

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

    def relabel(self, y, x):
        y_names = x["y_names"]
        for key in x:
            if key == "y_names":
                continue
            obj = x[key]

            if obj["type"] == "var-var":
                x_in_ynames = "{}:{}".format(obj["xScopeId"], obj["xName"])
                y_in_ynames = "{}:{}".format(obj["yScopeId"], obj["yName"])
                obj["xName"] = y[y_names.index(x_in_ynames)]
                obj["yName"] = y[y_names.index(y_in_ynames)]

            elif obj["type"] == "var-lit":
                x_in_ynames = "{}:{}".format(obj["xScopeId"], obj["xName"])
                obj["xName"] = y[y_names.index(x_in_ynames)]

    def remove_number(self, y):
        tmp = []
        for st in y:
            index = st.find(":")
            tmp.append(st[index+1:])
        return tmp

    def score(self, y, x):
        assert len(y) == len(x["y_names"]), \
            "two length should be equal, but len(y):{0}, len(x):{1}".format(
            len(y), len(x["y_names"])
        )
        y = self.remove_number(y)
        x = copy.deepcopy(x)
        self.relabel(y, x)


def parse_JSON(file_path):
    with open(file_path, "r") as f:
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
    return function_keys, programs


def main(args):
    function_keys, programs = parse_JSON(args.json)
    func = FeatureFucntion(function_keys)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train to get weight")
    parser.add_argument("-j", "--json", required=True, dest="json")
    # parser.add_argument("-o", "--output", required=True, dest="output")
    args = parser.parse_args()

    main(args)
