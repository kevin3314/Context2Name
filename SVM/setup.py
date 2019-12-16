import os
from distutils.core import setup

import numpy
from Cython.Build import cythonize
from setuptools import Extension

cpps = [
    os.path.join("../3rdparty/simdjson/src", x)
    for x in os.listdir("../3rdparty/simdjson/src")
    if x[-4:] == ".cpp"
]

print(cpps)

extesions = [
    Extension("simdjson", ["simdjson.pyx"] + cpps, extra_compile_args=["-std=c++14"])
]
setup(
    name="simdjson",
    ext_modules=cythonize(extesions),
    include_dirs=[numpy.get_include(), "../3rdparty/simdjson/include/", "../3rdparty/simdjson/src"],
)

extesions = [Extension("utils", ["utils.pyx"], extra_compile_args=["-std=c++11"])]
setup(
    name="utils",
    ext_modules=cythonize(extesions),
    include_dirs=[numpy.get_include(), "../3rdparty/simdjson/include"],
)

extesions = [Extension("SVM", ["SVM.pyx"], extra_compile_args=["-std=c++11"])]
setup(
    name="SVM",
    ext_modules=cythonize(extesions),
    include_dirs=[numpy.get_include(), "../3rdparty/simdjson/include"],
)
