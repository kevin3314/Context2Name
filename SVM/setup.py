#!/usr/bin/env python
import os
import os.path
import sys
import platform
from collections import namedtuple
from distutils.core import setup

import numpy
from Cython.Build import cythonize
from setuptools import Extension, find_packages
from setuptools.command.build_ext import build_ext


class BuildAssist(build_ext):
    """Allows overriding (or extending) the configuration for building C/C++
    extensions based on the compiler & platform.
    """
    # TODO: Keep fleshing this out and make it a separate python package, since
    # it would be useful to others.

    #: If CFLAGS (on gcc/clang) or CL (msvc) is defined in the environment
    #: variables then ignored extra_compile_args set by handlers.
    DO_NOT_REPLACE_CFLAGS = True

    #: If LDFLAGS (on gcc/clang) or LINK (msvc) is defined in the environment
    #: variables then ignored extra_link_args set by handlers.
    DO_NOT_REPLACE_LDFLAGS = True

    def build_extensions(self):
        # If a more precise handler is available, use it first. If it isn't,
        # try using a compiler-specific handler.
        compiler_handler = getattr(
            self,
            'using_{cc}_on_{platform}'.format(
                cc=self.compiler.compiler_type,
                platform=self.platform
            ),
            getattr(
                self,
                'using_{cc}'.format(
                    cc=self.compiler.compiler_type
                ),
                None
            )
        )
        if compiler_handler:
            try:
                result = compiler_handler(self.compiler)
            except NotImplementedError:
                print(
                    '[build_assist] No C/C++ handler found for {cc}'
                    ' on {platform}'.format(
                        cc=self.compiler.compiler_type,
                        platform=self.platform
                    )
                )
            else:
                if 'extra_compile_args' in result:
                    if (
                        self.DO_NOT_REPLACE_CFLAGS and
                        (os.environ.get('CFLAGS') or os.environ.get('CL'))
                    ):
                        print(
                            '[build_assist] CFLAGS/CL is set in environment,'
                            ' so default compiler arguments of {flags!r} will'
                            ' not be used.'.format(
                                flags=result['extra_compile_args']
                            )
                        )
                    else:
                        for e in self.extensions:
                            e.extra_compile_args = result['extra_compile_args']

                if 'extra_link_args' in result:
                    if (
                        self.DO_NOT_REPLACE_LDFLAGS and
                        (os.environ.get('LDFLAGS') or os.environ.get('LINK'))
                    ):
                        print(
                            '[build_assist] LDFLAGS/LINK is set in'
                            ' environment, so default linker arguments of'
                            ' {flags!r} will not be used.'.format(
                                flags=result['extra_link_args']
                            )
                        )
                    else:
                        for e in self.extensions:
                            e.extra_link_args = result['extra_link_args']

        super(BuildAssist, self).build_extensions()

    @property
    def platform(self):
        if sys.platform == 'win32':
            return 'windows'
        return os.uname()[0].lower()

    @property
    def darwin_version(self):
        # platform.mac_ver returns a result on platforms *other* than OS X,
        # make sure we're actually on it first...
        if self.platform == 'darwin':
            return namedtuple('mac_ver', ['major', 'minor'])(
                *(int(v) for v in platform.mac_ver()[0].split('.')[:2])
            )


class SIMDJsonBuild(BuildAssist):
    def using_msvc(self, compiler):
        return {
            'extra_compile_args': [
                '/std:c++17',
                '/arch:AVX2'
            ]
        }

    def using_unix(self, compiler):
        return {
            'extra_compile_args': [
                '-std=c++17',
                '-march=native'
            ]
        }

    def using_unix_on_darwin(self, compiler):
        if self.darwin_version.major >= 10 and self.darwin_version.minor >= 7:
            # After OS X Lion libstdc is deprecated, so we need to make sure we
            # link against libc++ instead.
            return {
                'extra_compile_args': [
                    '-march=native',
                    '-stdlib=libc++'
                ],
                'extra_link_args': [
                    '-lc++',
                    '-nodefaultlibs'
                ]
            }

        return {
            'extra_compile_args': [
                '-std=c++17',
                '-march=native'
            ]
        }


extesions = [
    Extension(
        "utils",
        sources=["utils.pyx"],
        extra_compile_args=["-std=c++17"],
        language='c++',
    )
]
setup(
    name="utils",
    ext_modules=cythonize(extesions),
    include_dirs=[numpy.get_include()],
    cmdclass={
        'build_ext': SIMDJsonBuild
    },
)

extesions = [
      Extension(
          "SVM",
          sources=["SVM.pyx"],
          extra_compile_args=["-std=c++17"],
          language='c++'
          )
      ]
setup(
      name="SVM",
      ext_modules=cythonize(extesions),
      include_dirs=[numpy.get_include()],
      cmdclass={
          'build_ext': SIMDJsonBuild
          },

      )
