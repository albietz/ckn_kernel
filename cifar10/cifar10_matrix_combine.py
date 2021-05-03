import argparse
import ckn_kernel
import numpy as np
import os
import time


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='compute cifar10 kernel matrix')
    parser.add_argument('--sub_block_size', type=int, default=100)
    parser.add_argument('--results_dir', default='data/cifar10_ckn_matrix')
    parser.add_argument('--model', default='exp_strided')
    parser.add_argument('--Ntrain', type=int, default=50000)
    parser.add_argument('--Ntest', type=int, default=10000)
    parser.add_argument('--remove_nans', action='store_true', help='remove files with NaNs')
    args = parser.parse_args()

    outdir = os.path.join(args.results_dir, args.model)
    sbsz = args.sub_block_size
    ntr = args.Ntrain // sbsz
    nte = args.Ntest // sbsz

    K = np.zeros((args.Ntrain, args.Ntrain), dtype=ckn_kernel.dtype)

    print('train')
    for i in range(ntr):
        print('.', end='', flush=True)
        for j in range(i + 1):
            fname = 'train_{}_{}.npy'.format(i, j)
            k = np.load(os.path.join(outdir, fname))
            if np.any(np.isnan(k)):
                print('warning:', fname, 'has NaNs')
                if args.remove_nans:
                    print('removing', fname)
                    os.remove(os.path.join(outdir, fname))

            sx = i * sbsz
            sy = j * sbsz
            K[sx:sx+sbsz,sy:sy+sbsz] = k

            if i != j:
                K[sy:sy+sbsz,sx:sx+sbsz] = k.T

    np.save(os.path.join(outdir, 'ktrain.npy'), K)

    print()
    print('test')
    K = np.zeros((args.Ntrain, args.Ntest), dtype=ckn_kernel.dtype)
    for i in range(ntr):
        print('.', end='', flush=True)
        for j in range(nte):
            fname = 'test_{}_{}.npy'.format(i, j)
            k = np.load(os.path.join(outdir, fname))
            if np.any(np.isnan(k)):
                print('warning:', fname, 'has NaNs')
                if args.remove_nans:
                    print('removing', fname)
                    os.remove(os.path.join(outdir, fname))

            sx = i * sbsz
            sy = j * sbsz
            K[sx:sx+sbsz,sy:sy+sbsz] = k

    np.save(os.path.join(outdir, 'ktest.npy'), K)
    print()
