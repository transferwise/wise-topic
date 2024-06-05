from matplotlib import pyplot as plt
import numpy as np


def plot_clustering(X_red, labels, title=None):
    x_min, x_max = np.min(X_red, axis=0), np.max(X_red, axis=0)
    X_red = (X_red - x_min) / (x_max - x_min)

    plt.figure(figsize=(6, 4))
    for label in np.unique(labels):
        plt.scatter(
            *X_red[labels == label].T,
            marker=f"${label}$",
            s=50,
            c=plt.cm.nipy_spectral(labels[labels == label] / 10),
            alpha=0.5,
        )

    plt.xticks([])
    plt.yticks([])
    if title is not None:
        plt.title(title, size=17)
    plt.axis("off")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
