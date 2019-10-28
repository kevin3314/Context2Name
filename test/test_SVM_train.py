import os
import sys
print(os.getcwd())
sys.path.append(os.getcwd())

from context2name.SVM_train import FeatureFucntion, parse_JSON

json_path = "./test.json"
function_keys, programs = parse_JSON(json_path)

test_key = set(["end", "t"])
test_ary = ["MemberExpression"]

def test_featurefunction_01():
    assert FeatureFucntion(function_keys).eval((test_key, test_ary)) == 1
