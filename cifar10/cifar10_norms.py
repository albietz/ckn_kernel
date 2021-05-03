import argparse
import ckn_kernel
import numpy as np
import os
import time

from cifar10_matrix import models

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='compute cifar10 kernel matrix')
    parser.add_argument('--data', default='data/cifar10_data/cifar_white.mat')
    parser.add_argument('--data_xtr', default='data/cifar10_data/cifar_white_xtrain.npy')
    parser.add_argument('--data_xte', default='data/cifar10_data/cifar_white_xtest.npy')
    parser.add_argument('--results_dir', default='data/cifar10_ckn_matrix')
    parser.add_argument('--model', default='exp_pool')
    parser.add_argument('--block_size', type=int, default=500)
    args = parser.parse_args()

    if args.model.endswith('_ntk'):
        use_ntk = True
        model_name = args.model[:-4]
    else:
        use_ntk = False
        model_name = args.model

    outdir = os.path.join(args.results_dir, args.model)
    os.makedirs(outdir, exist_ok=True)

    for test in [False, True]:
        print('test' if test else 'train')
        print('loading data...', end='', flush=True)
        # Xtr, Xte = read_dataset_cifar10_whitened(args.data)
        if test:
            X = np.load(args.data_xte).astype(ckn_kernel.dtype)
        else:
            X = np.load(args.data_xtr).astype(ckn_kernel.dtype)
        print('done')

        model = models[model_name]

        norms, norms_inv = [], []
        t = time.time()
        for i in range(0, X.shape[0], args.block_size):
            print('.', end='', flush=True)
            norms_blk, norms_inv_blk = ckn_kernel.compute_norms(X[i:(i+args.block_size)], model=model, ntk=use_ntk)
            norms.append(norms_blk)
            norms_inv.append(norms_inv_blk)
        print('done ({:.2f}s)'.format(time.time() - t))

        norms = np.concatenate(norms)
        assert not np.any(np.isnan(norms))
        norms_inv = np.concatenate(norms_inv)
        assert not np.any(np.isnan(norms_inv))
        suffix = '_test' if test else '_train'
        np.save(os.path.join(outdir, 'norms{}.npy'.format(suffix)), norms)
        np.save(os.path.join(outdir, 'norms_inv{}.npy'.format(suffix)), norms_inv)
