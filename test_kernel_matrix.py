import ckn_kernel, numpy as np

np.random.seed(43)
x = np.random.rand(5, 28, 28, 1).astype(np.float64)
model = [{'npatch': 3, 'subsampling': 2, 'sigma': 0.65},
         {'npatch': 3, 'subsampling': 5, 'sigma': 0.65}]
x1 = x[0]
print(ckn_kernel.compute_kernel(x1, x1, model))
x1 = x[1]
print(ckn_kernel.compute_kernel(x1, x1, model))
x1 = x[2]
print(ckn_kernel.compute_kernel(x1, x1, model))
x1 = x[3]
print(ckn_kernel.compute_kernel(x1, x1, model))
# print(ckn_kernel.compute_kernel(x1, x1, model))
# print(ckn_kernel.compute_kernel(x1, x1, model))
# print(ckn_kernel.compute_kernel(x1, x1, model))
