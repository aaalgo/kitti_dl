#!/usr/bin/env python3
import sys
import os
import subprocess as sp
import numpy
from distutils.core import setup, Extension

libraries = []
cv2libs = sp.check_output('pkg-config --libs opencv', shell=True).decode('ascii')
if 'opencv_imgcodecs' in cv2libs:
    libraries.append('opencv_imgcodecs')
    pass

numpy_include = os.path.join(os.path.abspath(os.path.dirname(numpy.__file__)), 'core', 'include')

if sys.version_info[0] < 3:
    boost_numpy = 'boost_numpy'
    boost_python = 'boost_python'
else:
    if os.path.exists('/usr/local/lib/libboost_python3.so'):
        boost_numpy = 'boost_numpy3'
        boost_python = 'boost_python3'
    else:
        boost_numpy = 'boost_numpy%d%d' % (sys.version_info[0], sys.version_info[1])
        boost_python = 'boost_python%d%d' % (sys.version_info[0], sys.version_info[1])
    pass

libraries.extend(['hdf5_hl', 'hdf5', 'boost_thread', 'boost_filesystem', 'boost_system', boost_numpy, boost_python, 'glog', 'gomp'])

cpp = Extension('cpp',
        language = 'c++',
        extra_compile_args = ['-O3', '-std=c++1y', '-g'], 
		include_dirs = ['/usr/lib/x86_64-linux-gnu/hdf5/serial/include', '/usr/local/include', numpy_include],
        libraries = libraries,
        library_dirs = ['/usr/lib/x86_64-linux-gnu/hdf5/serial/lib', '/usr/local/lib'],

        sources = ['python-api.cpp']
        )

setup (name = 'cpp',
       version = '0.0.1',
       author = 'Wei Dong',
       author_email = 'wdong@wdong.org',
       license = 'BSD',
       description = 'This is a demo package',
       ext_modules = [cpp],
       )
