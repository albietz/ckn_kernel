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

IF 1:
    ctypedef float float_t
    npDOUBLE = np.NPY_FLOAT32
    dtype = np.float32
ELSE:
    ctypedef double float_t
    npDOUBLE = np.NPY_FLOAT64
    dtype = np.float64

to_pool_type = {
    'gaussian': 0,
    'strided': 1,
    'average': 2,
}

to_kernel_type = {
    'exp': 0,
    'relu': 1,
    'linear': 2,
    'poly2': 3,
    'poly3': 4,
    'poly4': 5,
    'square': 6,
}

cdef extern from "CKNKernelMatrix.h":
    cdef cppclass CKNKernelMatrixEigen[Double]:
        Double computeKernel(const Double* const im1,
                             const Double* const im2,
                             const Double* const norms1,
                             const Double* const norms2,
                             const Double* const normsInv1,
                             const Double* const normsInv2,
                             const bool ntk,
                             const bool verbose) nogil

        Double cachedRFKernel()


    CKNKernelMatrixEigen[Double]* cknNew[Double](
                                    const size_t h,
                                    const size_t w,
                                    const size_t c,
                                    const vector[size_t]&,
                                    const vector[size_t]&,
                                    const vector[size_t]&,
                                    const vector[int]&,
                                    const vector[double]&,
                                    const vector[int]&,
                                    const bool) nogil

    Double computeAllKernel[Double](const Double* const im1,
                                    const Double* const im2,
                                    const bool ntk,
                                    const size_t h,
                                    const size_t w,
                                    const size_t c,
                                    const vector[size_t]&,
                                    const vector[size_t]&,
                                    const vector[size_t]&,
                                    const vector[int]&,
                                    const vector[double]&,
                                    const vector[int]&) nogil

    Double computeKernel[Double](const Double* const im1,
                                 const Double* const im2,
                                 const Double* const norms1,
                                 const Double* const norms2,
                                 const Double* const normsInv1,
                                 const Double* const normsInv2,
                                 const bool ntk,
                                 const size_t h,
                                 const size_t w,
                                 const size_t c,
                                 const vector[size_t]&,
                                 const vector[size_t]&,
                                 const vector[size_t]&,
                                 const vector[int]&,
                                 const vector[double]&,
                                 const vector[int]&) nogil

    Double computeNorms[Double](const Double* const im,
                                Double* norms,
                                Double* normsInv,
                                const size_t h,
                                const size_t w,
                                const size_t c,
                                const vector[size_t]&,
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
    cdef vector[size_t] pool_factors = [l.get('poolfactor', l['subsampling']) for l in model]
    cdef vector[int] kernel_types = [to_kernel_type[l.get('kernel', 'relu' if ntk else 'exp')] for l in model]
    cdef vector[double] kernel_params = [l.get('sigma', 1.0) for l in model]
    cdef vector[int] pools = [to_pool_type[l.get('pooling', 'gaussian')] for l in model]
    return computeAllKernel[double](&x1[0], &x2[0], ntk, h, w, c,
                                 patch_sizes, subs, pool_factors, kernel_types, kernel_params, pools)


@cython.boundscheck(False)
def compute_dist_to_ref(im_ref, ims, model, ntk=False):
    cdef size_t N, h, w, c
    h, w, c = im_ref.shape
    N = ims.shape[0]
    assert ims.shape[1:] == im_ref.shape

    cdef bool ntk_ = ntk
    cdef vector[size_t] patch_sizes = [l['npatch'] for l in model]
    cdef vector[size_t] subs = [l['subsampling'] for l in model]
    cdef vector[size_t] pool_factors = [l.get('poolfactor', l['subsampling']) for l in model]
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
            k00 = computeAllKernel[double](&x_ref[0], &x_ref[0], ntk_, h, w, c,
                                         patch_sizes, subs, pool_factors, kernel_types, kernel_params, pools)
        elif j % 2 == 0:
            kxy[j / 2] = computeAllKernel[double](&x_ref[0], &x[j / 2,0], ntk_, h, w, c,
                                         patch_sizes, subs, pool_factors, kernel_types, kernel_params, pools)
        else:
            kyy[j / 2] = computeAllKernel[double](&x[j / 2,0], &x[j / 2,0], ntk_, h, w, c,
                                         patch_sizes, subs, pool_factors, kernel_types, kernel_params, pools)

    return k00, np.asarray(kxy), np.asarray(kyy)


def compute_kernel_matrix(ims1, ims2=None, norms1=None, norms2=None,
                          norms_inv1=None, norms_inv2=None, model=None,
                          ntk=False, verbose=False):
    if norms1 is None or norms2 is None or norms_inv1 is None or norms_inv2 is None:
        return compute_kernel_matrix_all(ims1, ims2=ims2, model=model, ntk=ntk,
                                         verbose=verbose)
    else:
        return compute_kernel_matrix_norms(ims1, ims2, norms1, norms2, norms_inv1, norms_inv2,
                                           model=model, ntk=ntk, verbose=verbose)



@cython.boundscheck(False)
def compute_kernel_matrix_all(ims1, ims2=None, model=None, ntk=False, verbose=False):
    cdef bool sym = False
    if ims2 is None:
        ims2 = ims1
        sym = True
    cdef size_t N1, N2, h, w, c
    N1, h, w, c = ims1.shape
    N2 = ims2.shape[0]

    cdef bool ntk_ = ntk
    cdef bool verbose_ = verbose
    cdef vector[size_t] patch_sizes = [l['npatch'] for l in model]
    cdef vector[size_t] subs = [l['subsampling'] for l in model]
    cdef vector[size_t] pool_factors = [l.get('poolfactor', l['subsampling']) for l in model]
    cdef vector[int] kernel_types = [to_kernel_type[l.get('kernel', 'relu' if ntk else 'exp')] for l in model]
    cdef vector[double] kernel_params = [l.get('sigma', 1.0) for l in model]
    cdef vector[int] pools = [to_pool_type[l.get('pooling', 'gaussian')] for l in model]

    cdef float_t[:,::1] x1 = ims1.reshape(N1, -1).astype(dtype)
    cdef float_t[:,::1] x2 = ims2.reshape(N2, -1).astype(dtype)

    cdef float_t[:,::1] k = np.zeros((N1, N2), dtype=dtype)

    cdef int j, m, n
    for j in range(N1 * N2):
        m = j // N2
        n = j % N2
        if not sym or m <= n:  # skip symmetric entries
            k[m, n] = computeAllKernel[float_t](&x1[m,0], &x2[n,0], ntk_, h, w, c,
                                         patch_sizes, subs, pool_factors, kernel_types, kernel_params, pools,
                                         verbose_)

    if sym:  # fill symmetric entries
        for m in range(N1):
            for n in range(m):
                k[m, n] = k[n, m]
    return np.asarray(k)


@cython.boundscheck(False)
def compute_kernel_matrix_norms(ims1, ims2, norms1, norms2, norms_inv1, norms_inv2, model=None, ntk=False, verbose=False):
    cdef bool sym = False
    if ims2 is None:
        ims2 = ims1
        sym = True
    cdef size_t N1, N2, h, w, c
    N1, h, w, c = ims1.shape
    N2 = ims2.shape[0]

    cdef bool ntk_ = ntk
    cdef bool verbose_ = verbose
    cdef vector[size_t] patch_sizes = [l['npatch'] for l in model]
    cdef vector[size_t] subs = [l['subsampling'] for l in model]
    cdef vector[size_t] pool_factors = [l.get('poolfactor', l['subsampling']) for l in model]
    cdef vector[int] kernel_types = [to_kernel_type[l.get('kernel', 'relu' if ntk else 'exp')] for l in model]
    cdef vector[double] kernel_params = [l.get('sigma', 1.0) for l in model]
    cdef vector[int] pools = [to_pool_type[l.get('pooling', 'gaussian')] for l in model]

    cdef float_t[:,::1] x1 = ims1.reshape(N1, -1).astype(dtype)
    cdef float_t[:,::1] x2 = ims2.reshape(N2, -1).astype(dtype)
    cdef float_t[:,::1] n1 = norms1.reshape(N1, -1).astype(dtype)
    cdef float_t[:,::1] n2 = norms2.reshape(N2, -1).astype(dtype)
    cdef float_t[:,::1] ninv1 = norms_inv1.reshape(N1, -1).astype(dtype)
    cdef float_t[:,::1] ninv2 = norms_inv2.reshape(N2, -1).astype(dtype)

    cdef float_t[:,::1] k = np.zeros((N1, N2), dtype=dtype)

    cdef int j, m, n
    for j in range(N1 * N2):
        m = j // N2
        n = j % N2
        if not sym or m <= n:  # skip symmetric entries
            k[m, n] = computeKernel[float_t](
                    &x1[m,0], &x2[n,0], &n1[m,0], &n2[n,0], &ninv1[m,0], &ninv2[n,0],
                    ntk_, h, w, c,
                    patch_sizes, subs, pool_factors, kernel_types, kernel_params, pools,
                    verbose_ & (n == 0))

    if sym:  # fill symmetric entries
        for m in range(N1):
            for n in range(m):
                k[m, n] = k[n, m]
    return np.asarray(k)


@cython.boundscheck(False)
def compute_norms(ims, model=None, ntk=False, verbose=False):
    cdef bool sym = False
    cdef size_t L, N, h, w, c
    N, h, w, c = ims.shape
    L = len(model)

    cdef bool ntk_ = ntk
    cdef bool verbose_ = verbose
    cdef vector[size_t] patch_sizes = [l['npatch'] for l in model]
    cdef vector[size_t] subs = [l['subsampling'] for l in model]
    cdef vector[size_t] pool_factors = [l.get('poolfactor', l['subsampling']) for l in model]
    cdef vector[int] kernel_types = [to_kernel_type[l.get('kernel', 'relu' if ntk else 'exp')] for l in model]
    cdef vector[double] kernel_params = [l.get('sigma', 1.0) for l in model]
    cdef vector[int] pools = [to_pool_type[l.get('pooling', 'gaussian')] for l in model]

    cdef float_t[:,::1] x = ims.reshape(N, -1).astype(dtype)

    cdef float_t[:,::1] norms = np.zeros((N, L * h * w), dtype=dtype)
    cdef float_t[:,::1] norms_inv = np.zeros((N, L * h * w), dtype=dtype)


    cdef int j
    for j in prange(N, nogil=True):
        computeNorms[float_t](&x[j,0], &norms[j,0], &norms_inv[j,0], h, w, c,
                                     patch_sizes, subs, pool_factors, kernel_types, kernel_params, pools,
                                     verbose_)

    return np.asarray(norms), np.asarray(norms_inv)

cdef class CKNKernel:
    cdef CKNKernelMatrixEigen[float_t]* ckn

    def __cinit__(self, shape, model=None, ntk=False, verbose=False):
        cdef size_t h, w, c
        h, w, c = shape

        cdef bool verbose_ = verbose
        cdef vector[size_t] patch_sizes = [l['npatch'] for l in model]
        cdef vector[size_t] subs = [l['subsampling'] for l in model]
        cdef vector[size_t] pool_factors = [l.get('poolfactor', l['subsampling']) for l in model]
        cdef vector[int] kernel_types = [to_kernel_type[l.get('kernel', 'relu' if ntk else 'exp')] for l in model]
        cdef vector[double] kernel_params = [l.get('sigma', 1.0) for l in model]
        cdef vector[int] pools = [to_pool_type[l.get('pooling', 'gaussian')] for l in model]

        self.ckn = cknNew[float_t](h, w, c, patch_sizes, subs, pool_factors, kernel_types, kernel_params,
                          pools, verbose_);

    def __dealloc__(self):
        del self.ckn

    @cython.boundscheck(False)
    def compute_kernel_matrix(self, ims1, ims2, norms1, norms2, norms_inv1, norms_inv2, ntk=False, verbose=False, return_rf=False):
        cdef bool sym = False
        if ims2 is None:
            ims2 = ims1
            sym = True
        cdef size_t N1, N2, h, w, c
        N1, h, w, c = ims1.shape
        N2 = ims2.shape[0]

        cdef bool ntk_ = ntk
        cdef bool rf_ = ntk and return_rf
        cdef bool verbose_ = verbose
        cdef float_t[:,::1] x1 = ims1.reshape(N1, -1).astype(dtype)
        cdef float_t[:,::1] x2 = ims2.reshape(N2, -1).astype(dtype)
        cdef float_t[:,::1] n1 = norms1.reshape(N1, -1).astype(dtype)
        cdef float_t[:,::1] n2 = norms2.reshape(N2, -1).astype(dtype)
        cdef float_t[:,::1] ninv1 = norms_inv1.reshape(N1, -1).astype(dtype)
        cdef float_t[:,::1] ninv2 = norms_inv2.reshape(N2, -1).astype(dtype)

        cdef float_t[:,::1] k = np.zeros((N1, N2), dtype=dtype)
        cdef float_t[:,::1] kRF = np.zeros((N1, N2), dtype=dtype)

        cdef int j, m, n
        for j in range(N1 * N2):
            m = j // N2
            n = j % N2
            if not sym or m <= n:  # skip symmetric entries
                k[m, n] = self.ckn.computeKernel(
                        &x1[m,0], &x2[n,0], &n1[m,0], &n2[n,0],
                        &ninv1[m,0], &ninv2[n,0], ntk_, verbose_ & (n == 0))
                if rf_:
                    kRF[m, n] = self.ckn.cachedRFKernel()

        if sym:  # fill symmetric entries
            for m in range(N1):
                for n in range(m):
                    k[m, n] = k[n, m]
                    if rf_:
                        kRF[m, n] = k[n, m]

        if ntk and return_rf:
            return np.asarray(k), np.asarray(kRF)
        else:
            return np.asarray(k)
