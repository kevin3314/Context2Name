import os
import sys
import copy
print(os.getcwd())
sys.path.append(os.getcwd())

from context2name.SVM_train import FeatureFucntion, parse_JSON

json_path = "./test.json"
function_keys, programs = parse_JSON(json_path)

x = programs[0]

test_key = set(["end", "t"])
test_ary = ["MemberExpression"]

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


def test_featurefunction_eval():
    assert FeatureFucntion(function_keys).eval((test_key, test_ary)) == 1


def test_featurefunction_replace():
    func = FeatureFucntion(function_keys)
    y = func.remove_number(test_y)
    pro = copy.deepcopy(x)
    func.relabel(y, pro)
    assert pro["0-1"]["xName"] == "index"


def test_featurefunction_big_score():
    func = FeatureFucntion(function_keys)
    pro = copy.deepcopy(x)
    val = func.score(correct_y, pro)
    assert val == 6400.0


def test_featurefunction_min_score():
    func = FeatureFucntion(function_keys)
    pro = copy.deepcopy(x)
    val = func.score(test_y, pro)
    assert val == 2084.0
