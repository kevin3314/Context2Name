from distutils.core import setup

import numpy
from Cython.Build import cythonize
from setuptools import Extension

extesions = [Extension("SVM", ["SVM.pyx", "simdjson.pyx"], extra_compile_args=["-std=c++11"])]

setup(
    name="SVM",
    ext_modules=cythonize(extesions),
    include_dirs=[numpy.get_include(), "../3rdparty/simdjson/include"],
)
