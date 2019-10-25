# import os.path
# import operator
import argparse
import collections
import json
import pickle

import numpy as np


def main(args):
    with open(args.json, "r") as f:
        jsonData = json.load(f)
    print(jsonData)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train to get weight")
    parser.add_argument("-j", "--json", required=True, dest="json")
    # parser.add_argument("-o", "--output", required=True, dest="output")
    args = parser.parse_args()

    main(args)
