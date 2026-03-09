"""
Visualization utilities — training curves, confusion matrix, sample predictions.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def plot_training_history(history, save_path=None):
    """Plot loss and accuracy curves over training epochs."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Training History", fontsize=14, fontweight="bold")

    epochs = range(1, len(history["loss"]) + 1)

    ax1.plot(epochs, history["loss"], color="#e74c3c", linewidth=2)
    ax1.set_title("Loss over Epochs")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, [a * 100 for a in history["accuracy"]], color="#2ecc71", linewidth=2)
    ax2.set_title("Accuracy over Epochs")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
    plt.show()


def plot_confusion_matrix(y_true, y_pred, class_names=None, save_path=None):
    """Plot a confusion matrix heatmap."""
    num_classes = len(np.unique(y_true))
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax)

    if class_names:
        ax.set_xticks(range(num_classes))
        ax.set_yticks(range(num_classes))
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_yticklabels(class_names)

    for i in range(num_classes):
        for j in range(num_classes):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color=color, fontsize=8)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_sample_predictions(X, y_true, y_pred, n=16, save_path=None):
    """
    Display a grid of sample predictions vs ground truth.
    Correct predictions shown in green, wrong in red.
    """
    fig = plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(4, 4, figure=fig)
    fig.suptitle("Sample Predictions", fontsize=14, fontweight="bold")

    indices = np.random.choice(X.shape[1], n, replace=False)

    for i, idx in enumerate(indices):
        ax = fig.add_subplot(gs[i // 4, i % 4])
        img = X[:, idx].reshape(28, 28)
        ax.imshow(img, cmap="gray")
        ax.axis("off")

        true_label = y_true[idx]
        pred_label = y_pred[idx]
        color = "#2ecc71" if true_label == pred_label else "#e74c3c"
        ax.set_title(f"T:{true_label} P:{pred_label}", fontsize=9, color=color)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
