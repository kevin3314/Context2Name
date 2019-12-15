# distutils: language = c++
# cython: language_level=3

from libcpp.string cimport string
from libcpp cimport bool

cdef extern from "simdjson/parsedjson.h" namespace "simdjson":
    cdef cppclass ParsedJson:
        bool is_valid()
        # string get_error_message()

cdef extern from "<simdjson/parsedjsoniterator.h>" namespace "simdjson::Iterator":
    cdef cppclass BasicIterator[T]:
        size_t get_depth

cdef extern from "simdjson/jsonparser.h" namespace "simdjson":
    ParsedJson build_parsed_json(string)

def wrap_build_parsed_json(x):
    cdef string str_x = string(bytes(x))
    cdef ParsedJson pj = build_parsed_json(str_x)
    if not pj.is_valid():
        print("not valid!")

def test_parse_json():
    s = "short/0.json"
    wrap_build_parsed_json(s)
