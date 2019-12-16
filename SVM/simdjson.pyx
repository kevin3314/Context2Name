# distutils: language = c++
# cython: language_level=3
# cython: c_string_type=unicode, c_string_encoding=utf8

from libcpp.string cimport string
from libcpp cimport bool
import os

cdef extern from "simdjson/parsedjson.h" namespace "simdjson":
    cdef cppclass ParsedJson:
        bool is_valid()
        string get_error_message()
    cdef cppclass Iterator:
        pass

cdef extern from "simdjson/padded_string.h" namespace "simdjson":
    cdef cppclass padded_string:
        pass

# cdef extern from "<simdjson/parsedjsoniterator.h>" namespace "simdjson::Iterator":
#     cdef cppclass BasicIterator[T]:
#         size_t get_depth

cdef extern from "simdjson/jsonparser.h" namespace "simdjson":
    ParsedJson build_parsed_json(padded_string)

cdef extern from "simdjson/jsonioutil.h" namespace "simdjson":
    padded_string get_corpus(string)

def test_parse_json():
    cdef char *filename = "short/0.json"
    cdef padded_string p = get_corpus(filename)
    cdef ParsedJson pj = build_parsed_json(p)
    cdef Iterator pjk = Iterator(pj)
    if pj.is_valid():
        print("valid!!")
