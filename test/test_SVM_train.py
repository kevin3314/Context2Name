import os
import sys
import copy
import pytest
print(os.getcwd())
sys.path.append(os.getcwd())

from context2name.SVM_train import FeatureFucntion, parse_JSON

json_path = "./test.json"
function_keys, programs = parse_JSON(json_path)

x = programs[0]

test_key = set(["end", "t"])
test_ary = ["MemberExpression"]

candidates = set()
for program in programs:
    vals = FeatureFucntion.remove_number(program["y_names"])
    candidates.union(vals)

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


@pytest.fixture(scope="function", autouse=True)
def func():
    func = FeatureFucntion(function_keys, candidates)
    yield func


@pytest.fixture(scope="function", autouse=True)
def pro():
    pro = copy.deepcopy(x)
    yield pro


def test_featurefunction_eval(func):
    assert func.eval((test_key, test_ary)) == 1


def test_featurefunction_replace(func, pro):
    y = FeatureFucntion.remove_number(test_y)
    func.relabel(y, pro)
    assert pro["0-1"]["xName"] == "index"


def test_featurefunction_big_score(func, pro):
    val = func.score(correct_y, pro)
    assert val == 6400.0


def test_featurefunction_min_score(func, pro):
    val = func.score(test_y, pro)
    assert val == 2084.0


@pytest.mark.develop
def test_featurefunction_infer(func, pro):
    val = func.inference(pro)
    assert val == 2084.0
