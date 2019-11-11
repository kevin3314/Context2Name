import os
import sys
import copy
import pytest
print(os.getcwd())
sys.path.append(os.getcwd())

from context2name.SVM_train import FeatureFucntion, parse_JSON, DIVIDER

json_path = "./output"
function_keys, programs, candidates = parse_JSON(json_path)

_, ex, _ = parse_JSON("./output/0.json")

x = ex[0]

test_key = "i区((&&区url"

test_y = [
    "1:index",
    "1:array",
    "1:hoge",
    "2:url",
    "2:viw",
    "2:ignoreCache",
    "2:callback",
    "2:_ural",
    "2:a",
    "3:xhr",
    "4:gads",
]

correct_y = [
  "1:url",
  "1:index",
  "1:i",
  "2:url",
  "2:view",
  "2:ignoreCache",
  "2:callback",
  "2:_url",
  "2:i",
  "3:xhr",
  "4:xhr"
]


@pytest.fixture(scope="module", autouse=True)
def func():
    func = FeatureFucntion(function_keys, candidates)
    func.update_all_top_candidates(4)
    yield func


@pytest.fixture(scope="function", autouse=True)
def pro():
    pro = copy.deepcopy(x)
    yield pro


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
    y = FeatureFucntion.remove_number(test_y)
    func.relabel(y, pro)
    assert pro["0-52"]["xName"] == "index"


def test_featurefunction_big_score(func, pro):
    val = func.score(correct_y, pro)
    assert val > 10


def test_featurefunction_min_score(func, pro):
    val = func.score(test_y, pro)
    assert val > 10


def test_featurefunction_infer(func, pro):
    val = func.inference(pro)
    assert val is None


@pytest.mark.develop
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


@pytest.mark.develop
def test_featurefunction_top_candidates(func, pro, sequence, sequence_ano):
    key = "url" + DIVIDER + sequence + DIVIDER + "split"
    func.write_weight(key, 100)

    key = "url"
    val = func.top_candidates(key, sequence, 4)
    assert val[0] == 'split'
