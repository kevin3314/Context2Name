# distutils: language = c++
# cython: language_level=3

from libcpp.string cimport string


cdef extern from "simdjson/parsedjson.h" namespace "simdjson":
    cdef cppclass ParsedJson:
        pass

cdef extern from "simdjson/jsonparser.h" namespace "simdjson":
    ParsedJson build_parsed_json(string)
