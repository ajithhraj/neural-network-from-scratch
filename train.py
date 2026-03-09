"""
train.py — Train the neural network on MNIST and evaluate results.

Usage:
    python train.py
    python train.py --epochs 50 --lr 0.01 --batch_size 128
"""

import argparse
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.neural_network import NeuralNetwork
from src.data_utils import load_mnist
from src.visualize import plot_training_history, plot_confusion_matrix, plot_sample_predictions


def parse_args():
    parser = argparse.ArgumentParser(description="Train a Neural Network from Scratch on MNIST")
    parser.add_argument("--epochs",     type=int,   default=30,   help="Number of training epochs")
    parser.add_argument("--lr",         type=float, default=0.05, help="Learning rate")
    parser.add_argument("--batch_size", type=int,   default=64,   help="Mini-batch size")
    parser.add_argument("--hidden",     type=int,   nargs="+", default=[128, 64],
                        help="Hidden layer sizes (e.g. --hidden 256 128)")
    parser.add_argument("--save_model", type=str,   default="model",
                        help="Filename to save model weights (no extension)")
    parser.add_argument("--no_plot",    action="store_true", help="Skip generating plots")
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Load Data ─────────────────────────
    print("Loading MNIST dataset...")
    X_train, y_train, X_test, y_test = load_mnist(data_dir="data")
    print(f"  Train: {X_train.shape[1]} samples | Test: {X_test.shape[1]} samples\n")

    # ── Build Network ─────────────────────
    input_dim  = 784                           # 28×28 pixels
    output_dim = 10                            # digits 0–9
    layer_dims = [input_dim] + args.hidden + [output_dim]
    activations = ["relu"] * len(args.hidden) + ["softmax"]

    print(f"Architecture: {layer_dims}")
    print(f"Activations:  {activations}")
    print(f"Learning rate: {args.lr}  |  Batch size: {args.batch_size}  |  Epochs: {args.epochs}\n")

    nn = NeuralNetwork(layer_dims, activations, learning_rate=args.lr)

    # ── Train ─────────────────────────────
    print("Training...\n")
    nn.train(X_train, y_train, epochs=args.epochs,
             batch_size=args.batch_size, verbose=True)

    # ── Evaluate ──────────────────────────
    train_acc = nn.evaluate(X_train, y_train) * 100
    test_acc  = nn.evaluate(X_test,  y_test)  * 100
    print(f"\n{'='*45}")
    print(f"  Final Train Accuracy : {train_acc:.2f}%")
    print(f"  Final Test  Accuracy : {test_acc:.2f}%")
    print(f"{'='*45}\n")

    # ── Save Model ────────────────────────
    nn.save(args.save_model)

    # ── Plots ─────────────────────────────
    if not args.no_plot:
        os.makedirs("outputs", exist_ok=True)
        y_pred  = nn.predict(X_test)
        y_true  = np.argmax(y_test, axis=0)

        plot_training_history(nn.history, save_path="outputs/training_history.png")
        plot_confusion_matrix(y_true, y_pred,
                              class_names=[str(i) for i in range(10)],
                              save_path="outputs/confusion_matrix.png")
        plot_sample_predictions(X_test, y_true, y_pred,
                                save_path="outputs/sample_predictions.png")
        print("Plots saved to outputs/")


if __name__ == "__main__":
    main()
