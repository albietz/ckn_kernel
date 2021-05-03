import argparse
import ckn_kernel
import numpy as np
import os
import sys
import time


models = {
        'exp_pool': [
            {'npatch': 3, 'subsampling': 2, 'kernel': 'exp', 'sigma': 0.65, 'pooling': 'gaussian'},
            {'npatch': 3, 'subsampling': 5, 'kernel': 'exp', 'sigma': 0.65, 'pooling': 'gaussian'}],
        'exp_pool_nosub': [
            {'npatch': 3, 'subsampling': 1, 'poolfactor': 2, 'kernel': 'exp', 'sigma': 0.65, 'pooling': 'gaussian'},
            {'npatch': 6, 'subsampling': 1, 'poolfactor': 10, 'kernel': 'exp', 'sigma': 0.65, 'pooling': 'gaussian'}],
        'exp_sigma_06_pool': [
            {'npatch': 3, 'subsampling': 2, 'kernel': 'exp', 'sigma': 0.6, 'pooling': 'gaussian'},
            {'npatch': 3, 'subsampling': 5, 'kernel': 'exp', 'sigma': 0.6, 'pooling': 'gaussian'}],
        'exp_sigma_06_pool_2_2_2': [
            {'npatch': 3, 'subsampling': 2, 'kernel': 'exp', 'sigma': 0.6, 'pooling': 'gaussian'},
            {'npatch': 3, 'subsampling': 2, 'kernel': 'exp', 'sigma': 0.6, 'pooling': 'gaussian'},
            {'npatch': 3, 'subsampling': 2, 'kernel': 'exp', 'sigma': 0.6, 'pooling': 'gaussian'}],
        'exp_sigma_06_pool_5': [
            {'npatch': 3, 'subsampling': 5, 'kernel': 'exp', 'sigma': 0.6, 'pooling': 'gaussian'}],
        'exp_sigma_06_pool_5_nosub': [
            {'npatch': 3, 'subsampling': 1, 'poolfactor': 5, 'kernel': 'exp', 'sigma': 0.6, 'pooling': 'gaussian'}],
        'relu_pool_5': [
            {'npatch': 3, 'subsampling': 5, 'kernel': 'relu', 'pooling': 'gaussian'}],
        'exp_sigma_06_pool_10': [
            {'npatch': 3, 'subsampling': 10, 'kernel': 'exp', 'sigma': 0.6, 'pooling': 'gaussian'}],
        'exp_sigma_06_conv_5_pool_8': [
            {'npatch': 5, 'subsampling': 8, 'kernel': 'exp', 'sigma': 0.6, 'pooling': 'gaussian'}],
        'exp_sigma_06_conv_5_pool_8_nosub': [
            {'npatch': 5, 'subsampling': 1, 'poolfactor': 8, 'kernel': 'exp', 'sigma': 0.6, 'pooling': 'gaussian'}],
        'relu_conv_5_pool_8': [
            {'npatch': 5, 'subsampling': 8, 'kernel': 'relu', 'pooling': 'gaussian'}],
        'exp_sigma_06_conv_5_pool_5': [
            {'npatch': 5, 'subsampling': 5, 'kernel': 'exp', 'sigma': 0.6, 'pooling': 'gaussian'}],
        'exp_sigma_06_conv_5_pool_5_nosub': [
            {'npatch': 5, 'subsampling': 1, 'poolfactor': 5, 'kernel': 'exp', 'sigma': 0.6, 'pooling': 'gaussian'}],
        'exp_sigma_06_conv_5_pool_2': [
            {'npatch': 5, 'subsampling': 2, 'kernel': 'exp', 'sigma': 0.6, 'pooling': 'gaussian'}],
        'exp_sigma_06_conv_5_pool_2_nosub': [
            {'npatch': 5, 'subsampling': 1, 'poolfactor': 2, 'kernel': 'exp', 'sigma': 0.6, 'pooling': 'gaussian'}],
        'exp_sigma_06_conv_6_pool_8': [
            {'npatch': 6, 'subsampling': 8, 'kernel': 'exp', 'sigma': 0.6, 'pooling': 'gaussian'}],
        'exp_sigma_06_06_pool_nosub': [
            {'npatch': 3, 'subsampling': 1, 'poolfactor': 2, 'kernel': 'exp', 'sigma': 0.6, 'pooling': 'gaussian'},
            {'npatch': 6, 'subsampling': 1, 'poolfactor': 10, 'kernel': 'exp', 'sigma': 0.6, 'pooling': 'gaussian'}],
        'exp_sigma_055_pool': [
            {'npatch': 3, 'subsampling': 2, 'kernel': 'exp', 'sigma': 0.55, 'pooling': 'gaussian'},
            {'npatch': 3, 'subsampling': 5, 'kernel': 'exp', 'sigma': 0.55, 'pooling': 'gaussian'}],
        'exp_sigma_05_pool': [
            {'npatch': 3, 'subsampling': 2, 'kernel': 'exp', 'sigma': 0.5, 'pooling': 'gaussian'},
            {'npatch': 3, 'subsampling': 5, 'kernel': 'exp', 'sigma': 0.5, 'pooling': 'gaussian'}],
        'relu_pool': [
            {'npatch': 3, 'subsampling': 2, 'kernel': 'relu', 'pooling': 'gaussian'},
            {'npatch': 3, 'subsampling': 5, 'kernel': 'relu', 'pooling': 'gaussian'}],
        'exp_strided': [
            {'npatch': 3, 'subsampling': 2, 'kernel': 'exp', 'sigma': 0.65, 'pooling': 'strided'},
            {'npatch': 3, 'subsampling': 5, 'kernel': 'exp', 'sigma': 0.65, 'pooling': 'strided'}],

        ## 1-layer conv 6 (use whitened data!) pooling with different sizes
        'white6_exp_conv_6_pool_2': [
            {'npatch': 6, 'subsampling': 2, 'kernel': 'exp', 'sigma': 0.6, 'pooling': 'gaussian'}],
        'white6_exp_conv_6_pool_4': [
            {'npatch': 6, 'subsampling': 4, 'kernel': 'exp', 'sigma': 0.6, 'pooling': 'gaussian'}],
        'white6_exp_conv_6_pool_6': [
            {'npatch': 6, 'subsampling': 6, 'kernel': 'exp', 'sigma': 0.6, 'pooling': 'gaussian'}],
        'white6_exp_conv_6_pool_8': [
            {'npatch': 6, 'subsampling': 8, 'kernel': 'exp', 'sigma': 0.6, 'pooling': 'gaussian'}],
        'white6_exp_conv_6_pool_10': [
            {'npatch': 6, 'subsampling': 10, 'kernel': 'exp', 'sigma': 0.6, 'pooling': 'gaussian'}],

        ## exp/relu combined with lin and poly kernels
        'relu_lin_pool': [
            {'npatch': 3, 'subsampling': 2, 'kernel': 'relu', 'pooling': 'gaussian'},
            {'npatch': 3, 'subsampling': 5, 'kernel': 'linear', 'pooling': 'gaussian'}],
        'relu_quad_pool': [
            {'npatch': 3, 'subsampling': 2, 'kernel': 'relu', 'pooling': 'gaussian'},
            {'npatch': 3, 'subsampling': 5, 'kernel': 'poly2', 'pooling': 'gaussian'}],
        'relu_lin_lin_pool_2_2_2': [
            {'npatch': 3, 'subsampling': 2, 'kernel': 'relu', 'pooling': 'gaussian'},
            {'npatch': 3, 'subsampling': 2, 'kernel': 'linear', 'pooling': 'gaussian'},
            {'npatch': 3, 'subsampling': 2, 'kernel': 'linear', 'pooling': 'gaussian'}],
        'relu_quad_quad_pool_2_2_2': [
            {'npatch': 3, 'subsampling': 2, 'kernel': 'relu', 'pooling': 'gaussian'},
            {'npatch': 3, 'subsampling': 2, 'kernel': 'poly2', 'pooling': 'gaussian'},
            {'npatch': 3, 'subsampling': 2, 'kernel': 'poly2', 'pooling': 'gaussian'}],
        'exp_sigma_06_lin_pool_2_5': [
            {'npatch': 3, 'subsampling': 2, 'kernel': 'exp', 'sigma': 0.6, 'pooling': 'gaussian'},
            {'npatch': 3, 'subsampling': 5, 'kernel': 'linear', 'pooling': 'gaussian'}],
        'exp_sigma_06_quad_pool_2_5': [
            {'npatch': 3, 'subsampling': 2, 'kernel': 'exp', 'sigma': 0.6, 'pooling': 'gaussian'},
            {'npatch': 3, 'subsampling': 5, 'kernel': 'poly2', 'pooling': 'gaussian'}],
        'exp_sigma_06_square_pool_2_5': [
            {'npatch': 3, 'subsampling': 2, 'kernel': 'exp', 'sigma': 0.6, 'pooling': 'gaussian'},
            {'npatch': 3, 'subsampling': 5, 'kernel': 'square', 'pooling': 'gaussian'}],
        'exp_sigma_06_poly3_pool_2_5': [
            {'npatch': 3, 'subsampling': 2, 'kernel': 'exp', 'sigma': 0.6, 'pooling': 'gaussian'},
            {'npatch': 3, 'subsampling': 5, 'kernel': 'poly3', 'pooling': 'gaussian'}],
        'exp_sigma_06_poly4_pool_2_5': [
            {'npatch': 3, 'subsampling': 2, 'kernel': 'exp', 'sigma': 0.6, 'pooling': 'gaussian'},
            {'npatch': 3, 'subsampling': 5, 'kernel': 'poly4', 'pooling': 'gaussian'}],
        'exp_sigma_06_quad_quad_pool_2_2_2': [
            {'npatch': 3, 'subsampling': 2, 'kernel': 'exp', 'sigma': 0.6, 'pooling': 'gaussian'},
            {'npatch': 3, 'subsampling': 2, 'kernel': 'poly2', 'pooling': 'gaussian'},
            {'npatch': 3, 'subsampling': 2, 'kernel': 'poly2', 'pooling': 'gaussian'}],
        'exp_poly4_lin_pool_2_2_2': [
            {'npatch': 3, 'subsampling': 2, 'kernel': 'exp', 'sigma': 0.6, 'pooling': 'gaussian'},
            {'npatch': 3, 'subsampling': 2, 'kernel': 'poly4', 'pooling': 'gaussian'},
            {'npatch': 3, 'subsampling': 2, 'kernel': 'linear', 'pooling': 'gaussian'}],
        'exp_lin_poly4_pool_2_2_2': [
            {'npatch': 3, 'subsampling': 2, 'kernel': 'exp', 'sigma': 0.6, 'pooling': 'gaussian'},
            {'npatch': 3, 'subsampling': 2, 'kernel': 'linear', 'pooling': 'gaussian'},
            {'npatch': 3, 'subsampling': 2, 'kernel': 'poly4', 'pooling': 'gaussian'}],
        'exp_lin_lin_pool_2_2_2': [
            {'npatch': 3, 'subsampling': 2, 'kernel': 'exp', 'sigma': 0.6, 'pooling': 'gaussian'},
            {'npatch': 3, 'subsampling': 2, 'kernel': 'linear', 'pooling': 'gaussian'},
            {'npatch': 3, 'subsampling': 2, 'kernel': 'linear', 'pooling': 'gaussian'}],
        'exp_sigma_06_square_square_pool_2_2_2': [
            {'npatch': 3, 'subsampling': 2, 'kernel': 'exp', 'sigma': 0.6, 'pooling': 'gaussian'},
            {'npatch': 3, 'subsampling': 2, 'kernel': 'square', 'pooling': 'gaussian'},
            {'npatch': 3, 'subsampling': 2, 'kernel': 'square', 'pooling': 'gaussian'}],
        'quad_quad_pool_2_5': [
            {'npatch': 3, 'subsampling': 2, 'kernel': 'poly2', 'pooling': 'gaussian'},
            {'npatch': 3, 'subsampling': 5, 'kernel': 'poly2', 'pooling': 'gaussian'}],
        'quad_quad_quad_pool_2_2_2': [
            {'npatch': 3, 'subsampling': 2, 'kernel': 'poly2', 'pooling': 'gaussian'},
            {'npatch': 3, 'subsampling': 2, 'kernel': 'poly2', 'pooling': 'gaussian'},
            {'npatch': 3, 'subsampling': 2, 'kernel': 'poly2', 'pooling': 'gaussian'}],
        'exp_quad_conv_3_1_pool_2_5': [
            {'npatch': 3, 'subsampling': 2, 'kernel': 'exp', 'sigma': 0.6, 'pooling': 'gaussian'},
            {'npatch': 1, 'subsampling': 5, 'kernel': 'poly2', 'pooling': 'gaussian'}],
        'exp_quad_conv_3_5_pool_2_5': [
            {'npatch': 3, 'subsampling': 2, 'kernel': 'exp', 'sigma': 0.6, 'pooling': 'gaussian'},
            {'npatch': 5, 'subsampling': 5, 'kernel': 'poly2', 'pooling': 'gaussian'}],
        'exp_quad_conv_3_7_pool_2_5': [
            {'npatch': 3, 'subsampling': 2, 'kernel': 'exp', 'sigma': 0.6, 'pooling': 'gaussian'},
            {'npatch': 7, 'subsampling': 5, 'kernel': 'poly2', 'pooling': 'gaussian'}],
        'exp_poly3_conv_3_5_pool_2_5': [
            {'npatch': 3, 'subsampling': 2, 'kernel': 'exp', 'sigma': 0.6, 'pooling': 'gaussian'},
            {'npatch': 5, 'subsampling': 5, 'kernel': 'poly3', 'pooling': 'gaussian'}],
        'exp_poly4_conv_3_5_pool_2_5': [
            {'npatch': 3, 'subsampling': 2, 'kernel': 'exp', 'sigma': 0.6, 'pooling': 'gaussian'},
            {'npatch': 5, 'subsampling': 5, 'kernel': 'poly4', 'pooling': 'gaussian'}],
        'exp_poly4_conv_3_5_pool_2s_5': [
            {'npatch': 3, 'subsampling': 2, 'kernel': 'exp', 'sigma': 0.6, 'pooling': 'strided'},
            {'npatch': 5, 'subsampling': 5, 'kernel': 'poly4', 'pooling': 'gaussian'}],
        'quad_exp_pool_2_5': [
            {'npatch': 3, 'subsampling': 2, 'kernel': 'poly2', 'pooling': 'gaussian'},
            {'npatch': 3, 'subsampling': 5, 'kernel': 'exp', 'sigma': 0.6, 'pooling': 'gaussian'}],

        'exp_exp_conv_3_5_pool_2_5': [
            {'npatch': 3, 'subsampling': 2, 'kernel': 'exp', 'sigma': 0.6, 'pooling': 'gaussian'},
            {'npatch': 5, 'subsampling': 5, 'kernel': 'exp', 'sigma': 0.6, 'pooling': 'gaussian'}],
        'exp_exp_exp_conv_3_5_5_pool_2_2_2': [
            {'npatch': 3, 'subsampling': 2, 'kernel': 'exp', 'sigma': 0.6, 'pooling': 'gaussian'},
            {'npatch': 5, 'subsampling': 2, 'kernel': 'exp', 'sigma': 0.6, 'pooling': 'gaussian'},
            {'npatch': 5, 'subsampling': 2, 'kernel': 'exp', 'sigma': 0.6, 'pooling': 'gaussian'}],
        'exp_exp_conv_5_3_pool_2_5': [
            {'npatch': 5, 'subsampling': 2, 'kernel': 'exp', 'sigma': 0.6, 'pooling': 'gaussian'},
            {'npatch': 3, 'subsampling': 5, 'kernel': 'exp', 'sigma': 0.6, 'pooling': 'gaussian'}],
        'white5_exp_exp_conv_5_3_pool_2_5': [  # use white5 data (whitening with 5x5 patches)
            {'npatch': 5, 'subsampling': 2, 'kernel': 'exp', 'sigma': 0.6, 'pooling': 'gaussian'},
            {'npatch': 3, 'subsampling': 5, 'kernel': 'exp', 'sigma': 0.6, 'pooling': 'gaussian'}],
        'exp_exp_conv_5_5_pool_2_5': [
            {'npatch': 5, 'subsampling': 2, 'kernel': 'exp', 'sigma': 0.6, 'pooling': 'gaussian'},
            {'npatch': 5, 'subsampling': 5, 'kernel': 'exp', 'sigma': 0.6, 'pooling': 'gaussian'}],
    }

def read_dataset_cifar10_whitened(mat_file):
    """read cifar dataset from matlab whitened file.
    Available here: http://pascal.inrialpes.fr/data2/mairal/data/cifar_white.mat
    """
    from scipy.io import loadmat
    mat = loadmat(mat_file)

    def get_X(Xin):
        # HCWN -> NHWC
        return np.ascontiguousarray(Xin.astype(np.float64).reshape(32, 3, 32, -1).transpose(3, 0, 2, 1))

    return get_X(mat['Xtr']), get_X(mat['Xte'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='compute cifar10 kernel matrix')
    parser.add_argument('x_block', type=int)
    parser.add_argument('y_block', type=int)
    parser.add_argument('block_size', type=int)
    parser.add_argument('--sub_block_size', type=int, default=100)
    parser.add_argument('--data', default='data/cifar10_data/cifar_white.mat')
    parser.add_argument('--data_xtr', default='data/cifar10_data/cifar_white_xtrain.npy')
    parser.add_argument('--data_xte', default='data/cifar10_data/cifar_white_xtest.npy')
    parser.add_argument('--results_dir', default='data/cifar10_ckn_matrix')
    parser.add_argument('--model', default='exp_sigma_06_pool')
    parser.add_argument('--overwrite', action='store_true', help='overwrite files, even if they exist')
    parser.add_argument('--interactive', action='store_true', help='compute only, do not store')
    parser.add_argument('--compute_norms', action='store_true', help='compute own patch norms instead of loading')
    parser.add_argument('--verbose', action='store_true', help='verbose computation')
    parser.add_argument('--test', action='store_true',
                        help='train-test matrix instead of train only')
    args = parser.parse_args()

    if args.model.endswith('_ntk'):
        use_ntk = True
        model_name = args.model[:-4]
    else:
        use_ntk = False
        model_name = args.model

    outdir = os.path.join(args.results_dir, args.model)
    os.makedirs(outdir, exist_ok=True)

    print('loading data...', end='', flush=True)
    # Xtr, Xte = read_dataset_cifar10_whitened(args.data)
    Xtr = np.load(args.data_xtr).astype(ckn_kernel.dtype)
    if args.test:
        Xte = np.load(args.data_xte).astype(ckn_kernel.dtype)
    if not args.compute_norms:
        if use_ntk:
            norms_dir = outdir[:-4]
        else:
            norms_dir = outdir
        if not os.path.exists(os.path.join(norms_dir, 'norms_train.npy')):
            print('norm file missing. compute norms with cifar10_norms.py, or add option --compute_norms')
            sys.exit(1)
        norms = np.load(os.path.join(norms_dir, 'norms_train.npy'))
        norms_inv = np.load(os.path.join(norms_dir, 'norms_inv_train.npy'))
        if args.test:
            norms_test = np.load(os.path.join(norms_dir, 'norms_test.npy'))
            norms_inv_test = np.load(os.path.join(norms_dir, 'norms_inv_test.npy'))
    print('done')

    model = models[model_name]

    ckn = ckn_kernel.CKNKernel(Xtr.shape[1:], model=model, ntk=use_ntk)

    sbsz = args.sub_block_size
    num_sub_blocks = args.block_size // sbsz  # sub-blocks per block
    # indicies in terms of sub-blocks
    x_idx_start = num_sub_blocks * args.x_block
    y_idx_start = num_sub_blocks * args.y_block
    is_symmetric = (x_idx_start == y_idx_start)

    for i in range(num_sub_blocks):
        jmax = num_sub_blocks
        if is_symmetric and not args.test:
            jmax = i + 1
        for j in range(jmax):
            x_idx = x_idx_start + i
            y_idx = y_idx_start + j
            fname = '{}_{}_{}.npy'.format('test' if args.test else 'train', x_idx, y_idx)
            if not args.interactive and not args.overwrite and os.path.exists(os.path.join(outdir, fname)):
                continue   # skip, for besteffort

            sx = x_idx * sbsz
            sy = y_idx * sbsz

            data_x = Xtr[sx:sx+sbsz]
            if args.test:
                data_y = Xte[sy:sy+sbsz]
            else:
                data_y = Xtr[sy:sy+sbsz]

            if not args.compute_norms:
                norms_x = norms[sx:sx+sbsz]
                norms_inv_x = norms_inv[sx:sx+sbsz]
                if args.test:
                    norms_y = norms_test[sy:sy+sbsz]
                    norms_inv_y = norms_inv_test[sy:sy+sbsz]
                else:
                    norms_y = norms[sy:sy+sbsz]
                    norms_inv_y = norms_inv[sy:sy+sbsz]

            print('computing/saving sub-block ({}, {})...'.format(x_idx, y_idx), end='', flush=True)
            t = time.time()
            if args.compute_norms:
                K = ckn_kernel.compute_kernel_matrix(data_x, data_y,
                        model=model, ntk=use_ntk, verbose=args.verbose)
            else:
                K = ckn.compute_kernel_matrix(data_x, data_y,
                        norms_x, norms_y, norms_inv_x, norms_inv_y,
                        ntk=use_ntk, verbose=args.verbose, return_rf=use_ntk)
                if use_ntk:
                    K, K_RF = K
            if not args.interactive:
                np.save(os.path.join(outdir, fname), K)
            print('done ({:.2f}s)'.format(time.time() - t))
            if args.interactive:
                print(K)
                if use_ntk:
                    print('RF:')
                    print(K_RF)
