import ckn_kernel, numpy as np

ntk = False
np.random.seed(43)
x = np.random.rand(5, 28, 28, 1).astype(np.float64)
# model = [{'npatch': 3, 'subsampling': 2, 'sigma': 0.65},
#          {'npatch': 3, 'subsampling': 5, 'sigma': 0.65}]
model = [{'npatch': 3, 'subsampling': 2, 'kernel': 'relu', 'pooling': 'gaussian'},
        {'npatch': 3, 'subsampling': 5, 'kernel': 'relu', 'pooling': 'gaussian'}]
im_ref = np.ascontiguousarray(x[0])
ims = x[1:]

def assert_eq(a, b):
    assert abs(a - b) < 1e-5

k00, kxy, kyy = ckn_kernel.compute_dist_to_ref(im_ref, ims, model, ntk=ntk)
print(k00, kxy, kyy)

kk00 = ckn_kernel.compute_kernel(im_ref, im_ref, model, ntk=ntk)
kkxy = [ckn_kernel.compute_kernel(im_ref, np.ascontiguousarray(ims[i]), model, ntk=ntk) for i in range(ims.shape[0])]
kkyy = [ckn_kernel.compute_kernel(np.ascontiguousarray(ims[i]), np.ascontiguousarray(ims[i]), model, ntk=ntk) for i in range(ims.shape[0])]
print(kk00, kkxy, kkyy)

assert_eq(k00, kk00)
for i in range(ims.shape[0]):
    assert_eq(kxy[i], kkxy[i])
    assert_eq(kyy[i], kkyy[i])
