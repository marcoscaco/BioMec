from distutils.core import setup, Extension
import numpy
import os

# os.environ['CC'] = 'g++';
setup(name='pafprocess_ext', version='1.0',
    ext_modules=[
        Extension('_pafprocess', ['/content/drive/My Drive/Colab Notebooks/BioMec/tf_pose/pafprocess/pafprocess.cpp', '/content/drive/My Drive/Colab Notebooks/BioMec/tf_pose/pafprocess/pafprocess.i'],
                  swig_opts=['-c++'],
                  depends=["pafprocess.h"],
                  include_dirs=[numpy.get_include(), '/content/drive/My Drive/Colab Notebooks/BioMec/tf_pose/pafprocess/'])
    ],
    py_modules=[
        "pafprocess"
    ]
)
