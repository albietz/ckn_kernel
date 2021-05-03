import argparse
import numpy as np
import os
import pickle
from sklearn.feature_extraction import image

RAW_DATASET_FOLDER = 'data/cifar10_data/cifar-10-batches-py/'

def whiten_images(X, verbose=True, patch_size=3):
    '''X: images, shape (num_images, h, w, num_channels).'''
    h, w, c = X.shape[1:]
    for idx in range(X.shape[0]):
        if verbose and idx % 1000 == 0:
            print(idx)
        im = X[idx]
        p = image.extract_patches_2d(im, (patch_size, patch_size))
        if p.ndim < 4:
            p = p[:,:,:,None]
        p -= p.mean((1,2))[:,None,None,:]
        im = image.reconstruct_from_patches_2d(p, (h, w, c))
        p = image.extract_patches_2d(im, (patch_size, patch_size))
        p = p.reshape(p.shape[0], -1)

        cov = p.T.dot(p)
        s, U = np.linalg.eigh(cov)
        s[s <= 0] = 0
        s = np.sqrt(s)
        ind = s < 1e-8 * s.max()
        s[ind == False] = 1. / np.sqrt(s[ind == False])
        s[ind] = 0

        p = p.dot(U.dot(np.diag(s)).dot(U.T))
        p = p.reshape(p.shape[0], patch_size, patch_size, -1)
        X[idx] = image.reconstruct_from_patches_2d(p, (h, w, c))


def unpickle_cifar(f):
    assert os.path.exists(f), \
            'Need to download Cifar10 dataset from https://www.cs.toronto.edu/~kriz/cifar.html first!'
    with open(f, 'rb') as fo:
        return pickle.load(fo, encoding='bytes')


def read_dataset_cifar10_raw(folder):
    """read the pickle files provided from 
    https://www.cs.toronto.edu/~kriz/cifar.html
    and returns all data in one numpy array for training and one for testing"""

    n_batch = 10000
    n_train = 5 * n_batch
    n_test = n_batch

    # transpose to (n, h, w, channels)
    Xtr = np.empty((n_train, 32, 32, 3), dtype=np.float64)
    Ytr = np.empty(n_train, dtype=np.uint8)
    for i in range(1, 6):
        d = unpickle_cifar(os.path.join(folder, 'data_batch_{}'.format(i)))

        Xtr[(i-1)*n_batch:i*n_batch] = \
            d[b'data'].reshape(n_batch, 3, 32, 32).transpose(0, 2, 3, 1) / 255.0
        Ytr[(i-1)*n_batch:i*n_batch] = d[b'labels']

    d = unpickle_cifar(os.path.join(folder, 'test_batch'))
    Xte = np.ascontiguousarray(d[b'data'].astype(np.float64).reshape(n_batch, 3, 32, 32).transpose(0, 2, 3, 1))/255.0
    Yte = np.array(d[b'labels'], dtype=np.uint8)
    return Xtr, Ytr, Xte, Yte


def load_dataset_raw():
    return read_dataset_cifar10_raw(RAW_DATASET_FOLDER)


def save_whitened(data_dir, patch_size=3, verbose=True):
    print('loading raw cifar10...')
    X, y, Xte, yte = load_dataset_raw()
    if not os.path.exists(os.path.join(data_dir, 'cifar_white_ytrain.npy')):
        print('saving labels...')
        np.save(os.path.join(data_dir, 'cifar_white_ytrain.npy'), y)
        np.save(os.path.join(data_dir, 'cifar_white_ytest.npy'), yte)

    import preprocess
    print('whitening Xtrain...')
    preprocess.whiten_images(X, verbose=verbose, patch_size=patch_size)
    print('saving Xtrain...')
    np.save(os.path.join(data_dir, f'cifar_white{patch_size}_xtrain.npy'), X)
    print('whitening Xtest...')
    preprocess.whiten_images(Xte, verbose=verbose, patch_size=patch_size)
    print('saving Xtest...')
    np.save(os.path.join(data_dir, f'cifar_white{patch_size}_xtest.npy'), Xte)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='save preprocessed data')
    parser.add_argument('--data_dir', default='data/cifar10_data/')
    parser.add_argument('--patch_size', type=int, default=3)
    args = parser.parse_args()

    save_whitened(args.data_dir, patch_size=args.patch_size)
