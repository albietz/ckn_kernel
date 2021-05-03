# Experiments on the Cifar10 dataset

Scripts in this folder are for reproducing the full-kernel results in the paper:

A. Bietti. [On Approximation in Deep Convolutional Networks: a Kernel Perspective](https://arxiv.org/abs/2102.10032). arXiv, 2021.

Here are some steps to train a model on Cifar10:
* Download the Cifar10 python files from [here](https://www.cs.toronto.edu/~kriz/cifar.html).
* Use `preprocess.py` to obtained data files with appropriate ZCA/whitening pre-processing at the patch level.
* Add your model to the `models` dictionary in `cifar10_matrix.py`, in the following we will consider `exp_sigma_06_pool`.
* Run `python cifar10_norms.py --model exp_sigma_06_pool` to compute and save patch-level norms which are used for computing general kernel evaluations.
* Run `python cifar10_matrix.py <i> <j> <size> [--test] --model exp_sigma_06_pool` in a parallel fashion to compute and save blocks `K[i*size:(i+1)*size, j*size:(j+1)*size]` of either the training kernel matrix of the train-test kernel matrix. The option `--sub_block_size` controls the size of the blocks saved to disk.
* Run `python cifar10_matrix_combine.py --model exp_sigma_06_pool [--Ntrain <ntrain>]` to combine blocks into a single kernel matrix file.
* Run `python cifar10_ridge_eval.py --model exp_sigma_06_pool [--Ntrain <ntrain>]` to evaluate ridge regression models.
