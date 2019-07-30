# distutils: include_dirs = /scratch/clear/abietti/local/include

import cython
import numpy as np
cimport numpy as np

from cpython.ref cimport PyObject, Py_INCREF
from cython.operator cimport dereference as deref
from cython.parallel import prange
from libc.stdint cimport int32_t, int64_t
from libcpp cimport bool
from libcpp.string cimport string
from libcpp.vector cimport vector

np.import_array()

to_pool_type = {
    'gaussian': 0,
    'strided': 1,
    'average': 2,
}

to_kernel_type = {
    'exp': 0,
    'relu': 1,
}

cdef extern from "CKNKernelMatrix.h":
    Double computeKernel[Double](const Double* const im1,
                                 const Double* const im2,
                                 const bool ntk,
                                 const size_t h,
                                 const size_t w,
                                 const size_t c,
                                 const vector[size_t]&,
                                 const vector[size_t]&,
                                 const vector[int]&,
                                 const vector[double]&,
                                 const vector[int]&) nogil

def compute_kernel(im1, im2, model, ntk=False):
    h, w, c = im1.shape
    assert im1.shape == im2.shape
    cdef double[::1] x1 = im1.reshape(-1)
    cdef double[::1] x2 = im2.reshape(-1)

    cdef vector[size_t] patch_sizes = [l['npatch'] for l in model]
    cdef vector[size_t] subs = [l['subsampling'] for l in model]
    cdef vector[int] kernel_types = [to_kernel_type[l.get('kernel', 'relu' if ntk else 'exp')] for l in model]
    cdef vector[double] kernel_params = [l.get('sigma', 1.0) for l in model]
    cdef vector[int] pools = [to_pool_type[l.get('pooling', 'gaussian')] for l in model]
    return computeKernel[double](&x1[0], &x2[0], ntk, h, w, c,
                                 patch_sizes, subs, kernel_types, kernel_params, pools)


@cython.boundscheck(False)
def compute_dist_to_ref(im_ref, ims, model, ntk=False):
    cdef size_t N, h, w, c
    h, w, c = im_ref.shape
    N = ims.shape[0]
    assert ims.shape[1:] == im_ref.shape

    cdef bool ntk_ = ntk
    cdef vector[size_t] patch_sizes = [l['npatch'] for l in model]
    cdef vector[size_t] subs = [l['subsampling'] for l in model]
    cdef vector[int] kernel_types = [to_kernel_type[l.get('kernel', 'relu' if ntk else 'exp')] for l in model]
    cdef vector[double] kernel_params = [l.get('sigma', 1.0) for l in model]
    cdef vector[int] pools = [to_pool_type[l.get('pooling', 'gaussian')] for l in model]

    cdef double[::1] x_ref = im_ref.reshape(-1)
    cdef double[:,::1] x = ims.reshape(N, -1)

    cdef double k00
    cdef double[:] kxy = np.zeros(N)
    cdef double[:] kyy = np.zeros(N)

    cdef int j
    for j in prange(2 * N + 1, nogil=True):
        if j == 2 * N:
            k00 = computeKernel[double](&x_ref[0], &x_ref[0], ntk_, h, w, c,
                                         patch_sizes, subs, kernel_types, kernel_params, pools)
        elif j % 2 == 0:
            kxy[j / 2] = computeKernel[double](&x_ref[0], &x[j / 2,0], ntk_, h, w, c,
                                         patch_sizes, subs, kernel_types, kernel_params, pools)
        else:
            kyy[j / 2] = computeKernel[double](&x[j / 2,0], &x[j / 2,0], ntk_, h, w, c,
                                         patch_sizes, subs, kernel_types, kernel_params, pools)

    return k00, np.asarray(kxy), np.asarray(kyy)
