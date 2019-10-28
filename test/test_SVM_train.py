import os
import sys
print(os.getcwd())
sys.path.append(os.getcwd())

from context2name.SVM_train import FeatureFucntion, parse_JSON

json_path = "./test.json"
function_keys, programs = parse_JSON(json_path)

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

def test_featurefunction_eval():
    assert FeatureFucntion(function_keys).eval((test_key, test_ary)) == 1

def test_featurefunction_replace():
    func = FeatureFucntion(function_keys)
    y = func.remove_number(test_y)
    func.relabel(y, programs[0])
    assert programs[0]["0-1"]["xName"] == "index"
