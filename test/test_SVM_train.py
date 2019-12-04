import copy
import os
import sys

import numpy as np
import pytest

import SVM.utils as utils
from SVM.SVM import FeatureFucntion
from SVM.utils import DIVIDER, parse_JSON

print(os.getcwd())
sys.path.append(os.getcwd())

json_path = "./partial"
function_keys, parsed_programs, candidates, label_seq_dict = parse_JSON(json_path)

x_keys, ex, x_candidates, x_label_seq_dict = parse_JSON("./partial/107.json")

x = ex[0]

test_key = "t区.&%!区parts"

test_y = [
    "1区url",
    "2区d",
    "3区t",
    "4区t",
    "7区t",
    "1区pearts",
    "2区pearts",
    "3区pearts",
    "4区qeuestionIt",
    "4区eexclaimIt",
    "4区pearts",
    "7区reegex",
    "7区weithStache",
    "7区weithSpace",
    "7区pearts",
    "5区s",
    "6区s",
    "8区s",
    "9区s",
]

correct_y = [
    "1区t",
    "2区t",
    "3区t",
    "4区t",
    "7区t",
    "1区parts",
    "2区parts",
    "3区parts",
    "4区questionIt",
    "4区exclaimIt",
    "4区parts",
    "7区regex",
    "7区withStache",
    "7区withSpace",
    "7区parts",
    "5区s",
    "6区s",
    "8区s",
    "9区s",
]

PICKLES_PATH = "partial"


@pytest.fixture(scope="module", autouse=True)
def func():
    func = FeatureFucntion(function_keys, candidates, label_seq_dict)
    yield func


@pytest.fixture(scope="function")
def func_pretrain():
    func = FeatureFucntion.load_pickles(PICKLES_PATH)
    yield func


@pytest.fixture(scope="module", autouse=True)
def x_func():
    func = FeatureFucntion(x_keys, x_candidates, x_label_seq_dict)
    yield func


@pytest.fixture(scope="function", autouse=True)
def pro():
    pro = copy.deepcopy(x)
    yield pro


@pytest.fixture(scope="function", autouse=True)
def programs():
    programs = copy.deepcopy(parsed_programs)
    yield programs


@pytest.fixture(scope="module", autouse=True)
def sequence():
    sequence = "(("
    yield sequence


@pytest.fixture(scope="module", autouse=True)
def sequence_ano():
    sequence = "!"
    yield sequence


def test_featurefunction_eval(func):
    assert func.eval(test_key) == 1


def test_featurefunction_big_score(func, pro):
    val = func.score(correct_y, pro)
    assert val > 10


def test_featurefunction_min_score(func, pro):
    val = func.score(test_y, pro)
    assert val > 10


def test_featurefunction_infer(func, pro):
    val = func.inference(pro)
    assert val == pro["y_names"]


def test_featurefunction_infer_x_func(x_func, pro):
    val = x_func.inference(pro)
    assert val == pro["y_names"]


def test_featurefunction__update_label_seq_dict_highest(func, pro):
    partial = "t" + DIVIDER + ".&%!"
    func.write_weight(partial + DIVIDER + "parts", 3)
    assert func.label_seq_dict[partial][0][1] == "parts"


def test_featurefunction__update_label_seq_dict_lowest(func, pro):
    partial = "t" + DIVIDER + ".&%!"
    func.write_weight(partial + DIVIDER + "parts", -3)
    assert func.label_seq_dict[partial][-1][1] == "parts"


@pytest.mark.develop
def test_featurefunction_part_of_inference(func_pretrain, pro):
    # initialize y:answer
    y = []
    gen = utils.token_generator()
    for st in x["y_names"]:
        index = st.find(DIVIDER)
        y.append(st[: index + 1] + next(gen))
    utils.relabel(y, x)

    for i in range(len(x["y_names"])):
        variable = y[i]
        var_scope_id = utils.get_scopeid(variable)
        var_name = utils.get_varname(variable)
        candidates = set()
        edges = []
        connected_edges = []

        for key, edge in x.items():
            if key == "y_names":
                continue

            if edge["type"] == "var-var":
                if edge["xName"] == var_name and edge["xScopeId"] == int(var_scope_id):
                    edges.append(edge)
                    connected_edges.append(edge["yName"] + DIVIDER + edge["sequence"])

                elif edge["yName"] == var_name and edge["yScopeId"] == int(
                    var_scope_id
                ):
                    edges.append(edge)
                    connected_edges.append(edge["xName"] + DIVIDER + edge["sequence"])

            else:  # "var-lit"
                if edge["xName"] == var_name and edge["xScopeId"] == int(var_scope_id):
                    edges.append(edge)
                    connected_edges.append(edge["yName"] + DIVIDER + edge["sequence"])

        # score = score_edge + loss function(if not provided, loss=0)
        score_v = func_pretrain.score_edge(edges)

        for edge in connected_edges:
            if edge in func_pretrain.label_seq_dict.keys():
                for v in func_pretrain.label_seq_dict[edge][:8]:
                    candidates.add(v[1])

        print("change on {}".format(variable))
        for candidate in candidates:
            pre_edges = copy.deepcopy(edges)
            print(candidate)
            print(score_v)
            pre_label = y[i]
            pre_varname = utils.get_varname(pre_label)

            # check duplicate
            if utils.duplicate_check(y, var_scope_id, candidate):
                continue

            # temporaly relabel infered labels
            y[i] = var_scope_id + DIVIDER + candidate

            # relabel edges with new label
            utils.relabel_edges(edges, pre_varname, var_scope_id, candidate)

            new_score_v = func_pretrain.score_edge(edges)

            if new_score_v < score_v:  # when score is not improved
                y[i] = pre_label
                utils.relabel_edges(edges, candidate, var_scope_id, pre_varname)
                assert edges == pre_edges
            else:
                score_v = new_score_v
    print(y)
    assert False


def test_featurefunction_score_edge(func, pro):
    count = 0
    edges = []
    for key, rel in pro.items():
        count += 1
        if count > 10:
            break

        if key == "y_names":
            continue
        edges.append(rel)
    val = func.score_edge(edges)
    assert val == 10


def test_featurefunction_subgrad(func, programs):
    val = func.subgrad(
        programs,
        utils.simple_sequence(0.03),
        utils.naive_loss,
        iterations=30,
        save_dir=PICKLES_PATH,
    )

    assert val == [0, 1, 2]


def test_featurefunction_pretrained(func_pretrain, pro):
    val = func_pretrain.inference(pro)

    assert val == pro["y_names"]
