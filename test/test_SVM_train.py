import copy
import os
import sys
print(os.getcwd())
sys.path.append(os.getcwd())

import numpy as np
import pytest

import SVM.utils as utils
from SVM.SVM import FeatureFucntion
from SVM.utils import DIVIDER, parse_JSON, Triplet

print(os.getcwd())
sys.path.append(os.getcwd())

json_path = "./short"
function_keys, parsed_programs, candidates, label_seq_dict = parse_JSON(json_path)

x_keys, ex, x_candidates, x_label_seq_dict = parse_JSON("./sandbox/2.json")

x = iter(ex).__next__()

test_key = Triplet("t", ",%!", "parts")

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


def test_featurefunction__update_label_seq_dict_highest(func, pro):
    partial = "t" + DIVIDER + ",%!"
    key = Triplet("t", ",%!", "parts")
    func.write_weight(key, 3)
    assert func.label_seq_dict[partial][0][1] == "parts"


def test_featurefunction__update_label_seq_dict_lowest(func, pro):
    partial = "t" + DIVIDER + ",%!"
    key = Triplet("t", ",%!", "parts")
    func.write_weight(key, -3)
    assert func.label_seq_dict[partial][-1][1] == "parts"


def test_featurefunction__score_without_weight(x_func, pro):
    res = x_func.score(pro["y_names"], pro, without_weight=True)
    assert np.all(res)
