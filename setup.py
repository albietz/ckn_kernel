from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

setup(
    name = 'ckn_kernel',
    ext_modules = cythonize([Extension(
        'ckn_kernel',
        ['ckn_kernel.pyx'],
        language='c++',
        include_dirs=[numpy.get_include()],
        extra_compile_args=['-std=c++11', '-fopenmp'],
        extra_link_args=['-std=c++11', '-fopenmp', '-lglog'],
        )])
)
