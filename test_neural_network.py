"""
Unit tests for neural_network.py
Run with: pytest tests/
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.neural_network import NeuralNetwork, relu, sigmoid, softmax, cross_entropy_loss


# ── Activation Tests ──────────────────────────────────

def test_relu_positive():
    assert relu(np.array([3.0])) == 3.0

def test_relu_negative():
    assert relu(np.array([-1.0])) == 0.0

def test_sigmoid_zero():
    assert abs(sigmoid(np.array([0.0])) - 0.5) < 1e-6

def test_sigmoid_range():
    z = np.random.randn(100)
    s = sigmoid(z)
    assert np.all(s >= 0) and np.all(s <= 1)

def test_softmax_sums_to_one():
    z = np.random.randn(5, 10)
    s = softmax(z)
    col_sums = np.sum(s, axis=0)
    assert np.allclose(col_sums, 1.0)


# ── Network Init Tests ────────────────────────────────

def test_weight_shapes():
    nn = NeuralNetwork([4, 8, 3], ["relu", "softmax"])
    assert nn.params["W1"].shape == (8, 4)
    assert nn.params["W2"].shape == (3, 8)
    assert nn.params["b1"].shape == (8, 1)

def test_bias_init_zero():
    nn = NeuralNetwork([4, 8, 3], ["relu", "softmax"])
    assert np.all(nn.params["b1"] == 0)


# ── Forward Pass Tests ────────────────────────────────

def test_forward_output_shape():
    nn = NeuralNetwork([4, 8, 3], ["relu", "softmax"])
    X = np.random.randn(4, 100)
    AL, _ = nn._forward(X)
    assert AL.shape == (3, 100)

def test_forward_softmax_output_valid():
    nn = NeuralNetwork([4, 8, 3], ["relu", "softmax"])
    X = np.random.randn(4, 50)
    AL, _ = nn._forward(X)
    assert np.all(AL >= 0) and np.allclose(np.sum(AL, axis=0), 1.0)


# ── Loss Tests ────────────────────────────────────────

def test_cross_entropy_perfect_prediction():
    y_pred = np.array([[1.0], [0.0]])
    y_true = np.array([[1.0], [0.0]])
    loss = cross_entropy_loss(y_pred, y_true)
    assert loss < 1e-6

def test_cross_entropy_positive():
    y_pred = np.array([[0.7, 0.3], [0.3, 0.7]])
    y_true = np.array([[1.0, 0.0], [0.0, 1.0]])
    loss = cross_entropy_loss(y_pred, y_true)
    assert loss > 0


# ── Training Tests ────────────────────────────────────

def test_loss_decreases_over_training():
    np.random.seed(0)
    nn = NeuralNetwork([4, 8, 3], ["relu", "softmax"], learning_rate=0.1)
    X = np.random.randn(4, 200)
    labels = np.random.randint(0, 3, 200)
    y = np.zeros((3, 200))
    y[labels, np.arange(200)] = 1

    nn.train(X, y, epochs=50, verbose=False)
    assert nn.history["loss"][-1] < nn.history["loss"][0]

def test_predict_output_shape():
    nn = NeuralNetwork([4, 8, 3], ["relu", "softmax"])
    X = np.random.randn(4, 30)
    preds = nn.predict(X)
    assert preds.shape == (30,)
    assert np.all(preds >= 0) and np.all(preds < 3)
