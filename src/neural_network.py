"""
Neural Network from Scratch using NumPy
Author: Ajith Raj
Description: A fully connected feedforward neural network implemented
             using only NumPy — no PyTorch, no TensorFlow.
"""

import numpy as np


# ─────────────────────────────────────────
#  Activation Functions
# ─────────────────────────────────────────

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)

def softmax(z):
    e = np.exp(z - np.max(z, axis=0, keepdims=True))
    return e / np.sum(e, axis=0, keepdims=True)

ACTIVATIONS = {
    "relu":    (relu,    relu_derivative),
    "sigmoid": (sigmoid, sigmoid_derivative),
    "softmax": (softmax, None),
}


# ─────────────────────────────────────────
#  Loss Functions
# ─────────────────────────────────────────

def cross_entropy_loss(y_pred, y_true):
    m = y_true.shape[1]
    log_probs = -np.log(y_pred + 1e-8)
    loss = np.sum(y_true * log_probs) / m
    return loss

def binary_cross_entropy_loss(y_pred, y_true):
    m = y_true.shape[1]
    loss = -np.sum(
        y_true * np.log(y_pred + 1e-8) +
        (1 - y_true) * np.log(1 - y_pred + 1e-8)
    ) / m
    return loss


# ─────────────────────────────────────────
#  Neural Network Class
# ─────────────────────────────────────────

class NeuralNetwork:
    """
    A flexible feedforward neural network built from scratch with NumPy.

    Parameters
    ----------
    layer_dims : list of int
        Number of neurons in each layer, including input and output.
        e.g. [784, 128, 64, 10] → input:784, two hidden layers, output:10

    activations : list of str
        Activation function for each layer (excluding input).
        Supported: 'relu', 'sigmoid', 'softmax'

    learning_rate : float
        Step size for gradient descent.

    Example
    -------
    >>> nn = NeuralNetwork([784, 128, 10], ['relu', 'softmax'], learning_rate=0.01)
    >>> nn.train(X_train, y_train, epochs=100)
    """

    def __init__(self, layer_dims, activations, learning_rate=0.01):
        assert len(activations) == len(layer_dims) - 1, \
            "Number of activations must equal number of layers minus 1."

        self.layer_dims = layer_dims
        self.activations = activations
        self.lr = learning_rate
        self.params = {}
        self.history = {"loss": [], "accuracy": []}

        self._init_weights()

    def _init_weights(self):
        """He initialization for ReLU layers, Xavier for others."""
        np.random.seed(42)
        L = len(self.layer_dims)
        for l in range(1, L):
            fan_in = self.layer_dims[l - 1]
            if self.activations[l - 1] == "relu":
                scale = np.sqrt(2.0 / fan_in)          # He init
            else:
                scale = np.sqrt(1.0 / fan_in)           # Xavier init

            self.params[f"W{l}"] = np.random.randn(self.layer_dims[l], fan_in) * scale
            self.params[f"b{l}"] = np.zeros((self.layer_dims[l], 1))

    # ── Forward Pass ──────────────────────

    def _forward(self, X):
        cache = {"A0": X}
        A = X
        L = len(self.layer_dims) - 1

        for l in range(1, L + 1):
            W = self.params[f"W{l}"]
            b = self.params[f"b{l}"]
            Z = W @ A + b
            act_fn, _ = ACTIVATIONS[self.activations[l - 1]]
            A = act_fn(Z)
            cache[f"Z{l}"] = Z
            cache[f"A{l}"] = A

        return A, cache

    # ── Backward Pass ─────────────────────

    def _backward(self, y_true, cache):
        grads = {}
        L = len(self.layer_dims) - 1
        m = y_true.shape[1]

        # Output layer gradient
        AL = cache[f"A{L}"]
        dA = AL - y_true          # works for softmax+cross-entropy & sigmoid+BCE

        for l in reversed(range(1, L + 1)):
            Z = cache[f"Z{l}"]
            A_prev = cache[f"A{l-1}"]
            W = self.params[f"W{l}"]
            act_name = self.activations[l - 1]

            if act_name == "softmax":
                dZ = dA
            else:
                _, act_deriv = ACTIVATIONS[act_name]
                dZ = dA * act_deriv(Z)

            grads[f"dW{l}"] = (dZ @ A_prev.T) / m
            grads[f"db{l}"] = np.sum(dZ, axis=1, keepdims=True) / m
            dA = W.T @ dZ

        return grads

    # ── Weight Update ─────────────────────

    def _update_params(self, grads):
        L = len(self.layer_dims) - 1
        for l in range(1, L + 1):
            self.params[f"W{l}"] -= self.lr * grads[f"dW{l}"]
            self.params[f"b{l}"] -= self.lr * grads[f"db{l}"]

    # ── Train ─────────────────────────────

    def train(self, X, y, epochs=100, batch_size=64, verbose=True):
        """
        Train the network using mini-batch gradient descent.

        Parameters
        ----------
        X : np.ndarray, shape (n_features, n_samples)
        y : np.ndarray, shape (n_classes, n_samples)  — one-hot encoded
        epochs : int
        batch_size : int
        verbose : bool
        """
        m = X.shape[1]

        for epoch in range(1, epochs + 1):
            # Shuffle
            perm = np.random.permutation(m)
            X_s, y_s = X[:, perm], y[:, perm]

            epoch_loss = 0
            num_batches = 0

            for i in range(0, m, batch_size):
                Xb = X_s[:, i:i + batch_size]
                yb = y_s[:, i:i + batch_size]

                AL, cache = self._forward(Xb)
                loss = cross_entropy_loss(AL, yb)
                grads = self._backward(yb, cache)
                self._update_params(grads)

                epoch_loss += loss
                num_batches += 1

            avg_loss = epoch_loss / num_batches
            acc = self.evaluate(X, y)
            self.history["loss"].append(avg_loss)
            self.history["accuracy"].append(acc)

            if verbose and (epoch % 10 == 0 or epoch == 1):
                print(f"Epoch {epoch:>4}/{epochs}  |  Loss: {avg_loss:.4f}  |  Accuracy: {acc*100:.2f}%")

    # ── Predict ───────────────────────────

    def predict(self, X):
        """Returns predicted class indices."""
        AL, _ = self._forward(X)
        return np.argmax(AL, axis=0)

    def predict_proba(self, X):
        """Returns raw output probabilities."""
        AL, _ = self._forward(X)
        return AL

    # ── Evaluate ──────────────────────────

    def evaluate(self, X, y):
        """Returns accuracy as a float between 0 and 1."""
        preds = self.predict(X)
        labels = np.argmax(y, axis=0)
        return np.mean(preds == labels)

    # ── Save / Load ───────────────────────

    def save(self, filepath):
        """Save model weights to a .npz file."""
        np.savez(filepath, **self.params)
        print(f"Model saved to {filepath}.npz")

    def load(self, filepath):
        """Load model weights from a .npz file."""
        data = np.load(filepath)
        self.params = {k: data[k] for k in data}
        print(f"Model loaded from {filepath}")
