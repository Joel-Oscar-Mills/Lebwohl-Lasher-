from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np


ext_modules = [
    Extension(
        "LebwohlLasher_cython",
        ["LebwohlLasher_cython.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=['/openmp']=['-fopenmp'],
        extra_link_args=['-fopenmp']
    )
]

setup(name="LebwohlLasher_cython",
      ext_modules=cythonize(ext_modules))