import ckn_kernel, numpy as np

np.random.seed(43)
x = np.random.rand(5, 28, 28, 1).astype(np.float64)
# model = [{'npatch': 3, 'subsampling': 2, 'sigma': 0.65},
#          {'npatch': 3, 'subsampling': 5, 'sigma': 0.65}]
#
model = [
    {'npatch': 3, 'subsampling': 1, 'kernel': 'exp', 'sigma': 0.65, 'pooling': 'strided'},
    {'npatch': 3, 'subsampling': 1, 'kernel': 'exp', 'sigma': 0.65, 'pooling': 'strided'},
    {'npatch': 3, 'subsampling': 1, 'kernel': 'exp', 'sigma': 0.65, 'pooling': 'strided'},
    {'npatch': 3, 'subsampling': 1, 'kernel': 'exp', 'sigma': 0.65, 'pooling': 'strided'},
    {'npatch': 3, 'subsampling': 2, 'kernel': 'exp', 'sigma': 0.65, 'pooling': 'gaussian'}]
print(ckn_kernel.compute_kernel_matrix(x, model=model, verbose=True))
