import os
import sys
import copy
import pytest
import numpy as np
print(os.getcwd())
sys.path.append(os.getcwd())

from SVM.SVM import FeatureFucntion
from SVM.utils import parse_JSON, DIVIDER
import SVM.utils as utils

json_path = "./short"
function_keys, parsed_programs, candidates, label_seq_dict = parse_JSON(json_path)

x_keys, ex, x_candidates, x_label_seq_dict = parse_JSON("./short/2.json")

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
    "9区s"
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
    "9区s"
]


@pytest.fixture(scope="function", autouse=True)
def pro():
    pro = copy.deepcopy(x)
    yield pro


def test_featurefunction_replace(pro):
    utils.relabel(test_y, pro)
    assert pro["0"]["xName"] == "url"


def test_featurefunction_duplicate_check(pro):
    boo = utils.duplicate_check(correct_y, 1, "t")
    assert boo


def test_featurefunction_duplicate_check2(pro):
    boo = utils.duplicate_check(correct_y, 1, "hogehoge")
    assert not boo
