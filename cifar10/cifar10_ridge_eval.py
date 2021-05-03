import argparse
import numpy as np
import scipy.linalg
import os
import sys
import time


def load_matrix(outdir, ntrain, ntest=10000, train_only=False, no_save=False):
    print('loading train matrix..')
    train_fname = os.path.join(outdir, 'ktrain_{}_{}.npy'.format(ntrain, ntrain))
    if os.path.exists(train_fname):
        Ktrain = np.load(train_fname)
    else:
        Ktrain = np.load(os.path.join(outdir, 'ktrain.npy'))
        assert Ktrain.shape[0] >= ntrain
        Ktrain = Ktrain[:ntrain,:ntrain]
        if not no_save and ntrain < 50000:
            np.save(train_fname, Ktrain)
    if train_only:
        return Ktrain

    print('loading test matrix..')
    test_fname = os.path.join(outdir, 'ktest_{}_{}.npy'.format(ntrain, ntest))
    if os.path.exists(test_fname):
        Ktest = np.load(test_fname)
    else:
        Ktest = np.load(os.path.join(outdir, 'ktest.npy'))
        assert Ktrain.shape[0] >= ntrain
        Ktest = Ktest[:ntrain,:ntest]
        if not no_save and ntrain < 50000:
            np.save(test_fname, Ktest)
    return Ktrain, Ktest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='compute cifar10 kernel matrix')
    parser.add_argument('--results_dir', default='data/cifar10_ckn_matrix')
    parser.add_argument('--model', default='exp_sigma_06_pool')
    parser.add_argument('--more_models', default='', help='list of models and weights model1:w1,model2:w2')
    parser.add_argument('--Ntrain', type=int, default=50000)
    parser.add_argument('--Ntest', type=int, default=10000)
    parser.add_argument('--M', type=int, default=None)
    parser.add_argument('--Nsmall', type=int, default=None)
    parser.add_argument('--load_only', action='store_true')
    parser.add_argument('--no_save', action='store_true')
    args = parser.parse_args()

    outdir = os.path.join(args.results_dir, args.model)

    print('loading labels')
    ytr = np.load('data/cifar10_data/cifar_white_ytrain.npy')[:args.Ntrain]
    yte = np.load('data/cifar10_data/cifar_white_ytest.npy')[:args.Ntest]

    y = np.zeros((args.Ntrain, 10))
    y[np.arange(args.Ntrain), ytr] = 1.
    y -= 0.1

    Ktrain, Ktest = load_matrix(outdir, ntrain=args.Ntrain, ntest=args.Ntest)
    if args.more_models:
        for mw in args.more_models.split(','):
            model, weight = mw.split(':')

            Ktr, Kte = load_matrix(os.path.join(args.results_dir, model), ntrain=args.Ntrain, ntest=args.Ntest)
            Ktrain += float(weight) * Ktr
            Ktest += float(weight) * Kte

    if args.load_only:
        sys.exit(0)

    I = np.eye(args.Ntrain)

    for lmbda in 10. ** np.arange(-4, -12, -1):
        print(lmbda)
        if args.M is not None:
            alpha = scipy.linalg.solve(Ktrain[:,:args.M].T.dot(Ktrain[:,:args.M]) + args.Ntrain * lmbda * Ktrain[:args.M,:args.M], Ktrain[:,:args.M].T.dot(y))
            yhat = Ktest[:args.M,:].T.dot(alpha)
        else:
            alpha = scipy.linalg.solve(Ktrain + args.Ntrain * lmbda * I, y)
            yhat = Ktest.T.dot(alpha)
        ypred = yhat.argmax(axis=1)
        print('accuracy:', np.mean(ypred == yte))
