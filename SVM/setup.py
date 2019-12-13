from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(name="SVM", ext_modules=cythonize("SVM.pyx"), include_dirs=[numpy.get_include()])
