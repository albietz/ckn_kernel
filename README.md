# Convolutional kernel computation (CKN, NTK)

Package for computing exact kernel evaluations for some convolutional kernels, such as those from convolutional kernel networks (CKNs, see [here](https://arxiv.org/abs/1406.3332), [here](https://arxiv.org/abs/1605.06265), [here](http://jmlr.org/papers/v20/18-190.html)) and neural tangent kernels for convolutional networks (NTK or CNTK, see [here](https://arxiv.org/abs/1806.07572), [here](https://arxiv.org/abs/1905.12173), [here](https://arxiv.org/abs/1904.11955)).
The main code is in C++, with Cython bindings.


## Compilation

The compilation requires Cython and glog (`sudo apt-get install libgoogle-glog-dev`). You can compile the python package with the following command:

```
python setup.py build_ext -if
```

## Examples

Here is a commented example for a two-layer CKN with exponential kernels on patches.

```python
import numpy as np
import ckn_kernel

# data (e.g. MNIST digits)
X = np.random.rand(5, 28, 28, 1).astype(np.float64)

# define the model:
#   2 convolutional layers
#   3x3 patches at both layers
#   pooling/downsampling by a factor 2 at the first layer, 5 at the second
#   patch kernels are exp((u - 1) / sigma^2)
model = [{'npatch': 3, 'subsampling': 2, 'pooling': 'gaussian', 'kernel': 'exp', 'sigma': 0.65},
         {'npatch': 3, 'subsampling': 5, 'pooling': 'gaussian', 'kernel': 'exp', 'sigma': 0.65}]

# compute (symmetric) kernel matrix
K = ckn_kernel.compute_kernel_matrix(X, model=model)

# for computing non-symmetric blocks, e.g. train/test, with test data Xtest,
# or simply for parallelizing different block computations, use:
# K_train_test = ckn_kernel.compute_kernel_matrix(X, Xtest, model=model)
```

The NTK can be used as follows (only the ReLU / arc-cosine kernel is supported for now):

```python
# model: same architecture as above, but with arc-cosine kernels (i.e. the dual of the ReLU activation)
model = [{'npatch': 3, 'subsampling': 2, 'pooling': 'gaussian', 'kernel': 'relu'},
         {'npatch': 3, 'subsampling': 5, 'pooling': 'gaussian', 'kernel': 'relu'}]

# compute NTK kernel matrix
K = ckn_kernel.compute_kernel_matrix(X, model=model, ntk=True)
```
Setting `ntk=False` in this last command would give the kernel corresponding to training only the last layer of a ReLU CNN at infinite width, which then corresponds to a more basic CKN.

For pooling, the options are `gaussian`, `average` or `strided`, where the latter corresponds to strided convolutions (i.e. downsampling with no blurring) and can be much faster, though it may induce aliasing.

## Papers

The papers to cite are the following (first one for CKN, second one for NTK):

A. Bietti, J. Mairal. [Group Invariance, Stability to Deformations, and Complexity of Deep Convolutional Representations](http://jmlr.org/papers/v20/18-190.html). JMLR, 2019.

A. Bietti, J. Mairal. [On the Inductive Bias of Neural Tangent Kernels](https://arxiv.org/abs/1905.12173). arXiv, 2019.
