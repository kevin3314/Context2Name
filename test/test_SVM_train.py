import os
import sys
import copy
import pytest
import numpy as np
print(os.getcwd())
sys.path.append(os.getcwd())

from context2name.SVM_train import FeatureFucntion
from context2name.utils import parse_JSON, DIVIDER
import context2name.utils as utils

json_path = "./output"
function_keys, parsed_programs, candidates = parse_JSON(json_path)

x_keys, ex, x_candidates = parse_JSON("./output/0.json")

x = ex[0]

test_key = "i区((&&区url"

test_y = [
    "1区index",
    "1区array",
    "1区hoge",
    "2区url",
    "2区viw",
    "2区ignoreCache",
    "2区callback",
    "2区_ural",
    "2区a",
    "3区xhr",
    "4区gads",
]

correct_y = [
  "1区url",
  "1区index",
  "1区i",
  "2区url",
  "2区view",
  "2区ignoreCache",
  "2区callback",
  "2区_url",
  "2区i",
  "3区xhr",
  "4区xhr"
]

WEIGHT_PATH = "tmp_weight"


@pytest.fixture(scope="module", autouse=True)
def func():
    func = FeatureFucntion(function_keys, candidates)
    func.update_all_top_candidates(4)
    yield func


@pytest.fixture(scope="module", autouse=True)
def func_pretrain():
    func = FeatureFucntion(function_keys, candidates, weight_path=WEIGHT_PATH+".npy")
    func.update_all_top_candidates(4)
    yield func


@pytest.fixture(scope="module", autouse=True)
def x_func():
    func = FeatureFucntion(x_keys, x_candidates)
    func.update_all_top_candidates(4)
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


def test_featurefunction_replace(func, pro):
    y = utils.remove_number(test_y)
    utils.relabel(y, pro)
    assert pro["0-52"]["xName"] == "index"


def test_featurefunction_big_score(func, pro):
    val = func.score(correct_y, pro)
    assert val > 10


def test_featurefunction_min_score(func, pro):
    val = func.score(test_y, pro)
    assert val > 10


def test_featurefunction_duplicate_check(func, pro):
    boo = utils.duplicate_check(correct_y, 1, "url")
    assert boo


def test_featurefunction_duplicate_check2(func, pro):
    boo = utils.duplicate_check(correct_y, 1, "hogehoge")
    assert not boo


@pytest.mark.develop
def test_featurefunction_infer(func, pro):
    val = func.inference(pro)
    assert val == ["aaa"]


def test_featurefunction_infer_x_func(x_func, pro):
    val = x_func.inference(pro)
    assert val == ["aaa"]


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


def test_featurefunction_top_candidates(func, pro, sequence, sequence_ano):
    key = "url" + DIVIDER + sequence + DIVIDER + "split"
    func.write_weight(key, 100)

    key = "url"
    val = func.top_candidates(key, sequence, 4)
    assert val[0] == 'split'


#@pytest.mark.develop
def test_featurefunction_subgrad(func, programs):
    val = func.subgrad(programs, utils.simple_sequence(0.03), utils.naive_loss, save_weight=WEIGHT_PATH)

    assert val == [0, 1, 2]


@pytest.mark.develop
def test_featurefunction_pretrained(func_pretrain, pro):
    val = func_pretrain.inference(pro)

    assert val == ["aaa"]
