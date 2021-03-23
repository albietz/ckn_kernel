import ckn_kernel, numpy as np
import sys

np.random.seed(43)
x = np.random.rand(2, 28, 28, 1).astype(np.float64)
model = [{'npatch': 3, 'subsampling': 2, 'sigma': 0.6, 'pooling': 'gaussian', 'kernel': 'exp'},
        {'npatch': 3, 'subsampling': 5, 'sigma': 0.6, 'pooling': 'gaussian', 'kernel': 'exp'}]

print(ckn_kernel.compute_kernel_matrix(x, model=model, verbose=True))

norms, norms_inv = ckn_kernel.compute_norms(x, model=model, verbose=True)

ckn = ckn_kernel.CKNKernel(x[0].shape, model=model, verbose=True)
print(ckn.compute_kernel_matrix(x, x, norms, norms, norms_inv, norms_inv))
