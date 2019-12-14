# distutils: language = c++
# cython: language_level=3

# import os.path
# import operator
import argparse
import bisect
import collections
import copy
import json
import os
import pickle
import time
from multiprocessing import Pool
from functools import partial

import numpy as np
cimport numpy as np
from tqdm import tqdm
from os.path import join

import utils as utils

from libcpp.string cimport string

cdef extern from "simdjson/jsonparser.h":
    pass

DIVIDER = "区"

DEF NUM_PATH = 20
DEF TOP_CANDIDATES = 16

cdef class FeatureFucntion:
    """Class for feature function.

    Attributes:
        function_keys : list :
            function_keys is list of feature.
            feature like: "id区((||区i"

        weight : np.ndarray :
            weight is weight to be learned.

        candidates_dict : dictionary:
            top S candidate dict.
            key is like "id区((||"

        candidates : LB :
            candidates of variable name.
    """

    cdef:
        dict function_keys,candidates,label_seq_dict
        np.ndarray weight

    property weight:
        def __get__(self):
            return self.weight

        def __set__(self, newval):
            self.weight = newval
            self._update_label_seq_dict()

    def __init__(self, function_keys, candidates, label_seq_dict):
        self.function_keys = function_keys
        self.candidates = candidates
        self.label_seq_dict = label_seq_dict
        self.weight = np.ones(len(function_keys))
        self._update_label_seq_dict()

    def _update_label_seq_dict(self):
        # sort __label_seq_dict with weight value
        for key, value in self.label_seq_dict.items():
            # each value is (index, label)
            value.sort(key=lambda x: self.weight[x[0]], reverse=True)

    def eval(self, key, without_weight=False):
        if key in self.function_keys:
            index = self.function_keys[key]
            if without_weight:
                return index
            else:
                return self.weight[index]
        return 0

    def write_weight(self, key, value):
        if key in self.function_keys:
            index = self.function_keys[key]
            self.weight[index] = value
            self._update_label_seq_dict()

    cpdef inference(self, x, loss=utils.dummy_loss, NUM_PATH=NUM_PATH, TOP_CANDIDATES=TOP_CANDIDATES):
        """inference program properties.
        x : program
        loss : loss function
        """
        cdef:
            list y, edges, connected_edges
            int iter_n, length_y_names, i, var_scope_id
            int x_scope_id, y_scope_id
            int score_v, new_score_v
            unicode key, type_label, var_name
            unicode x_name, y_name

        # initialize y:answer
        y = []
        x = copy.deepcopy(x)
        gen = utils.token_generator()

        y = [f"{utils.get_scopeid(st)}{DIVIDER}{next(gen)}" for st in x["y_names"]]
        utils.relabel(y, x)

        length_y_names = len(x["y_names"])
        for iter_n in range(NUM_PATH):
            # each node with unknown property in the G^x
            for i in range(length_y_names):
                variable = y[i]
                var_scope_id = int(utils.get_scopeid(variable))
                var_name = utils.get_varname(variable)
                candidates = set()
                edges = []
                connected_edges = []

                for key_tmp, edge in x.items():
                    key = key_tmp
                    if key == "y_names":
                        continue

                    type_label = edge["type"]
                    if type_label == "var-var":
                        x_name = edge["xName"]
                        y_name = edge["yName"]
                        x_scope_id = edge["xScopeId"]
                        y_scope_id = edge["yScopeId"]
                        if (
                            x_name == var_name
                            and x_scope_id == var_scope_id
                        ):
                            edges.append(edge)
                            connected_edges.append(
                                edge["yName"] + DIVIDER + edge["sequence"]
                            )

                        elif (
                            y_name == var_name
                            and y_scope_id == var_scope_id
                        ):
                            edges.append(edge)
                            connected_edges.append(
                                edge["xName"] + DIVIDER + edge["sequence"]
                            )

                    else:  # "var-lit"
                        x_name = edge["xName"]
                        x_scope_id = edge["xScopeId"]
                        if (
                            x_name == var_name
                            and x_scope_id == var_scope_id
                        ):
                            edges.append(edge)
                            connected_edges.append(
                                edge["yName"] + DIVIDER + edge["sequence"]
                            )

                # score = score_edge + loss function(if not provided, loss=0)
                score_v = self.score_edge(edges) + loss(x["y_names"], y)

                for edge in connected_edges:
                    if edge in self.label_seq_dict.keys():
                        for v in self.label_seq_dict[edge][:TOP_CANDIDATES]:
                            candidates.add(v[1])

                if not candidates:
                    continue

                for candidate in candidates:
                    pre_label = y[i]
                    pre_name = utils.get_varname(pre_label)
                    # check duplicate
                    if utils.duplicate_check(y, var_scope_id, candidate):
                        continue

                    # relabel edges with new label
                    utils.relabel_edges(
                        edges, pre_name, var_scope_id, candidate)

                    # temporaly relabel infered labels
                    y[i] = str(var_scope_id) + DIVIDER + candidate
                    x["y_names"][i] = y[i]

                    # score = score_edge + loss
                    new_score_v = self.score_edge(
                        edges) + loss(x["y_names"], y)

                    if new_score_v < score_v:  # when score is not improved
                        y[i] = pre_label
                        x["y_names"][i] = pre_label
                        utils.relabel_edges(edges, candidate, var_scope_id, pre_name)

        return y

    def inference_only_correct_number(self, program, **kwrags):
        y = self.inference(program, **kwrags)
        val = 0
        for a, b in zip(program["y_names"], y):
            if a == b:
                val += 1
        return val, len(y)

    def score(self, y, x, without_weight=False):
        assert len(y) == len(
            x["y_names"]
        ), "two length should be equal, but len(y):{0}, len(x):{1}".format(
            len(y), len(x["y_names"])
        )
        x = copy.deepcopy(x)
        utils.relabel(y, x)
        if without_weight:
            res = np.zeros(len(self.function_keys))
        else:
            res = 0
        for key in x:
            if key == "y_names":
                continue
            obj = x[key]
            x_name = obj["xName"]
            y_name = obj["yName"]
            seq = obj["sequence"]
            key_name = x_name + DIVIDER + seq + DIVIDER + y_name
            val = self.eval(key_name, without_weight=without_weight)

            if not val:
                continue

            if without_weight:
                res[val] += 1
            else:
                res += val
        return res

    def score_edge(self, edges):
        res = 0
        for edge in edges:
            x_name = edge["xName"]
            y_name = edge["yName"]
            seq = edge["sequence"]
            key_name = x_name + DIVIDER + seq + DIVIDER + y_name
            res += self.eval(key_name)
        return res

    def subgrad_mmsc(self, program, loss, only_loss=False):
        # this default g value may be wrong
        y_i = program["y_names"]
        y_star = self.inference(program, loss)
        sum_loss = (
            self.score(y_star, program) + loss(y_star, y_i) -
            self.score(y_i, program)
        )
        if only_loss:
            return sum_loss

        g = (self.score(y_star, program, without_weight=True) - self.score(y_i, program, without_weight=True))
        label_loss = loss(y_star, y_i)
        return g, sum_loss, label_loss

    def subgrad(self, programs, stepsize_sequence, loss_function, *, using_norm=False, iterations=30, save_dir=None, LAMBDA=0.5, BETA=0.5, init_weight_proportion=0.5, verbose=True, profile=False):
        def calc_l2_norm(weight):
            return np.linalg.norm(weight, ord=2) / 2 * LAMBDA

        # initialize
        weight_zero = np.ones(len(self.function_keys)) * (BETA * init_weight_proportion)
        self.weight = weight_zero
        weight_t = weight_zero
        learning_rate = next(stepsize_sequence)
        pre_sum_wrong_label = None

        # best loss, weight
        best_loss = float('inf')
        best_weight = weight_zero

        for i in tqdm(range(iterations)):
            # get newest weight
            sum_loss = 0

            # calculate grad
            subgrad_with_loss = partial(self.subgrad_mmsc, loss=loss_function)
            if profile:
                res = list(tqdm(map(subgrad_with_loss, programs), total=len(programs)))
            else:
                with Pool() as pool:
                    res = list(tqdm(pool.imap_unordered(subgrad_with_loss, programs), total=len(programs)))

            grad, sum_loss, sum_wrong_label = (sum(x) for x in zip(*res))

            grad /= len(programs)
            sum_loss /= len(programs)

            if using_norm:
                sum_loss += calc_l2_norm(weight_t)

            if sum_loss < best_loss:
                best_loss = sum_loss
                best_weight = weight_t

            new_weight = utils.projection(
                weight_t - learning_rate * grad, 0, BETA
            )

            if pre_sum_wrong_label and pre_sum_wrong_label < sum_wrong_label:
                print("not improvement! iteration={}".format(i))
                learning_rate = next(stepsize_sequence)
            pre_sum_wrong_label = sum_wrong_label

            self.weight = new_weight
            weight_t = new_weight

            if verbose:
                print(best_weight[:50])

        sum_loss = 0
        # calculate loss for last weight
        subgrad_with_only_loss = partial(self.subgrad_mmsc, loss=loss_function, only_loss=True)
        with Pool() as pool:
            res = pool.map(subgrad_with_only_loss, programs)

        sum_loss = sum(res)
        sum_loss /= len(programs)
        if using_norm:
            sum_loss += calc_l2_norm(self.weight)

        # return weight for min loss
        if sum_loss < best_loss:
            best_loss = sum_loss
            best_weight = weight_t

        self.weight = best_weight
        if save_dir:
            self._make_pickles(save_dir)
        return best_weight

    def _make_pickles(self, save_dir):
        with open(join(save_dir, "svm.pickle"), mode="wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load_pickles(save_dir):
        with open(join(save_dir, "svm.pickle"), mode="rb") as f:
            svm = pickle.load(f)
        return svm


def main(args):
    function_keys, programs, candidates, label_seq_dict = utils.parse_JSON(args.input_dir)
    func = FeatureFucntion(function_keys, candidates, label_seq_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train to get weight")
    parser.add_argument("-i", "--input", required=True, dest="input_dir")
    # parser.add_argument("-o", "--output", required=True, dest="output")
    args = parser.parse_args()

    main(args)
