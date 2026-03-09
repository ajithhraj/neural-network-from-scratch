"""
Data utilities — load and preprocess MNIST dataset.
"""

import numpy as np
import os
import urllib.request
import gzip
import struct


def download_mnist(data_dir="data"):
    """Download raw MNIST binary files if not already present."""
    os.makedirs(data_dir, exist_ok=True)
    base_url = "http://yann.lecun.com/exdb/mnist/"
    mirror   = "https://ossci-datasets.s3.amazonaws.com/mnist/"

    files = [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz",
    ]

    for fname in files:
        fpath = os.path.join(data_dir, fname)
        if not os.path.exists(fpath):
            url = mirror + fname
            print(f"Downloading {fname} ...")
            urllib.request.urlretrieve(url, fpath)

    print("All MNIST files ready.")


def _load_images(path):
    with gzip.open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(n, rows * cols)
    return images


def _load_labels(path):
    with gzip.open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels


def load_mnist(data_dir="data"):
    """
    Load and preprocess MNIST.

    Returns
    -------
    X_train : np.ndarray, shape (784, 60000)  — pixel values in [0, 1]
    y_train : np.ndarray, shape (10, 60000)   — one-hot encoded labels
    X_test  : np.ndarray, shape (784, 10000)
    y_test  : np.ndarray, shape (10, 10000)
    """
    download_mnist(data_dir)

    X_train = _load_images(os.path.join(data_dir, "train-images-idx3-ubyte.gz"))
    y_train = _load_labels(os.path.join(data_dir, "train-labels-idx1-ubyte.gz"))
    X_test  = _load_images(os.path.join(data_dir, "t10k-images-idx3-ubyte.gz"))
    y_test  = _load_labels(os.path.join(data_dir, "t10k-labels-idx1-ubyte.gz"))

    # Normalize pixel values to [0, 1]
    X_train = X_train.T / 255.0
    X_test  = X_test.T  / 255.0

    # One-hot encode labels
    y_train = one_hot(y_train, 10)
    y_test  = one_hot(y_test,  10)

    return X_train, y_train, X_test, y_test


def one_hot(labels, num_classes):
    """Convert integer labels to one-hot encoded matrix."""
    m = labels.shape[0]
    one_hot_matrix = np.zeros((num_classes, m))
    one_hot_matrix[labels, np.arange(m)] = 1
    return one_hot_matrix


def train_test_split(X, y, test_ratio=0.2, seed=42):
    """Simple train/test split for custom datasets."""
    np.random.seed(seed)
    m = X.shape[1]
    perm = np.random.permutation(m)
    split = int(m * (1 - test_ratio))
    train_idx, test_idx = perm[:split], perm[split:]
    return X[:, train_idx], y[:, train_idx], X[:, test_idx], y[:, test_idx]
