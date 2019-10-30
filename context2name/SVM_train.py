# import os.path
# import operator
import argparse
import collections
import copy
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

    def __init__(self, function_keys, candidates):
        self.function_keys = function_keys
        self.weight = np.ones(len(function_keys))
        self.candidates = candidates

    def eval(self, key):
        if key in self.function_keys:
            index = self.function_keys.index(key)
            return self.weight[index]
        return 0

    @classmethod
    def relabel(cls, y, x):
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

    @classmethod
    def remove_number(cls, y):
        tmp = []
        for st in y:
            index = st.find(":")
            tmp.append(st[index + 1:])
        return tmp

    def inference(self, x):
        # initialize y:answer
        y = []
        for st in x["y_names"]:
            index = st.find(":")
            y.append(st[: index + 1] + "i")
        num_path = 10  # the number of iterations.
        for i in range(num_path):
            # each node with unknown property in the G^x
            for variable in x["y_names"]:
                index = variable.find(":")
                var_scope_id = int(variable[:index])
                var_name = variable[index+1:]
                rels = []

                for key, rel in x.items():
                    if key == "y_names":
                        continue

                    if rel["type"] == "var-var":
                        if (rel["xName"] == var_name and rel["xScopeId"] == var_scope_id) or \
                                (rel["yName"] == var_name and rel["yScopeId"] == var_scope_id):
                            rels.append(rel)
                    else:  # "var-lit"
                        if (rel["xName"] == var_name and rel["xScopeId"] == var_scope_id):
                            rels.append(rel)

    def score(self, y, x):
        assert len(y) == len(x["y_names"]), \
            "two length should be equal, but len(y):{0}, len(x):{1}".format(
                len(y), len(x["y_names"])
            )
        y = self.remove_number(y)
        x = copy.deepcopy(x)
        self.relabel(y, x)
        val = 0
        for key in x:
            if key == "y_names":
                continue
            obj = x[key]
            k = set([obj["xName"], obj["yName"]])
            seq = obj["sequence"]
            val += self.eval((k, seq))
        return val

    def score_edge(self, edges):
        res = 0
        for edge in edges:
            var_key = set([edge["xName"], edge["yName"]])
            var_seq = edge["sequence"]
            res += self.eval((var_key, var_seq))
        return res



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

    candidates = set()
    for program in programs:
        vals = remove_number(program["y_names"])
        candidates = candidates.union(vals)

    func = FeatureFucntion(function_keys, candidates)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train to get weight")
    parser.add_argument("-j", "--json", required=True, dest="json")
    # parser.add_argument("-o", "--output", required=True, dest="output")
    args = parser.parse_args()

    main(args)
