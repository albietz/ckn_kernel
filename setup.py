from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy
import os

eigen_dir = os.path.join('include', 'Eigen')
if not os.path.exists(eigen_dir):
    print('Downloading Eigen...')

    from glob import glob
    # from urllib.request import urlretrieve
    import requests
    import shutil
    import tarfile

    if not os.path.exists('include'):
        os.mkdir('include')
    eigen_url = 'https://gitlab.com/libeigen/eigen/-/archive/3.3.9/eigen-3.3.9.tar.gz'
    tar_path = os.path.join('include', 'Eigen.tar.gz')
    r = requests.get(eigen_url)
    with open(tar_path, 'wb') as outfile:
        outfile.write(r.content)
    # urlretrieve(eigen_url, tar_path)
    with tarfile.open(tar_path, 'r') as tar:
        def is_within_directory(directory, target):
        	
        	abs_directory = os.path.abspath(directory)
        	abs_target = os.path.abspath(target)
        
        	prefix = os.path.commonprefix([abs_directory, abs_target])
        	
        	return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
        	for member in tar.getmembers():
        		member_path = os.path.join(path, member.name)
        		if not is_within_directory(path, member_path):
        			raise Exception("Attempted Path Traversal in Tar File")
        
        	tar.extractall(path, members, numeric_owner=numeric_owner) 
        	
        
        safe_extract(tar, "include")
    thedir = glob(os.path.join('include', 'eigen-*'))[0]
    shutil.move(os.path.join(thedir, 'Eigen'), eigen_dir)
    shutil.move(os.path.join(thedir, 'unsupported'), os.path.join('include', 'unsupported'))
    print('done!')


libs = ['mkl_sequential', 'mkl_core', 'mkl_intel_lp64', 'iomp5']
# libs = ['mkl_rt', 'iomp5']
conda_include = os.path.join(os.environ['CONDA_PREFIX'], 'include')
setup(
    name = 'ckn_kernel',
    ext_modules = cythonize([Extension(
        'ckn_kernel',
        ['ckn_kernel.pyx'],
        language='c++',
        libraries=libs,
        include_dirs=[conda_include, numpy.get_include(), 'include'],
        # extra_compile_args=['-std=c++11', '-DEIGEN_USE_MKL_ALL', '-DAXPBY', '-fopenmp']
        extra_compile_args=['-std=c++11', '-DAXPBY', '-fopenmp']
        )])
)
