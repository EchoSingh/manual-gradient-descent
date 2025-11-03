"""
Simple batch gradient descent linear regression implementation.

Created by: Aditya Singh
Date: 2025-11-03

API:
 - fit(X, y, lr=0.01, epochs=1000, verbose=False) -> (theta, history)
 - predict(X, theta) -> predictions
 - compute_mse(X, y, theta) -> mse

This implementation uses NumPy and does not rely on scikit-learn.
"""
from typing import Tuple, Dict
import numpy as np


def _ensure_array(x):
    arr = np.asarray(x, dtype=float)
    return arr


def add_intercept(X: np.ndarray) -> np.ndarray:
    X = _ensure_array(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    ones = np.ones((X.shape[0], 1))
    return np.hstack((ones, X))


def predict(X: np.ndarray, theta: np.ndarray) -> np.ndarray:
    Xb = add_intercept(X)
    return Xb.dot(theta)


def compute_mse(X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> float:
    y = _ensure_array(y)
    preds = predict(X, theta)
    errors = preds - y
    return float(np.mean(errors ** 2))


def fit(X: np.ndarray, y: np.ndarray, lr: float = 0.01, epochs: int = 1000, verbose: bool = False) -> Tuple[np.ndarray, Dict]:
    """Fit linear regression using batch gradient descent.

    Args:
        X: shape (n_samples,) or (n_samples, n_features)
        y: shape (n_samples,)
        lr: learning rate
        epochs: number of gradient steps
        verbose: print progress

    Returns:
        theta: parameters shape (n_features+1,)
        history: dict with 'mse' list
    """
    Xb = add_intercept(X)
    y = _ensure_array(y)
    n_samples, n_features = Xb.shape

    theta = np.zeros(n_features, dtype=float)
    history = {"mse": []}

    for epoch in range(epochs):
        preds = Xb.dot(theta)
        error = preds - y
        grad = (2.0 / n_samples) * Xb.T.dot(error)
        theta -= lr * grad
        mse = float(np.mean(error ** 2))
        history["mse"].append(mse)
        if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
            print(f"Epoch {epoch+1}/{epochs}: mse={mse:.6f}")

    return theta, history


if __name__ == "__main__":
    # quick local demo
    import matplotlib.pyplot as plt

    rng = np.random.RandomState(0)
    X = rng.rand(100) * 10
    y = 3.0 * X + 2.0 + rng.randn(100) * 2.0

    theta, history = fit(X, y, lr=0.001, epochs=5000, verbose=True)
    print("learned theta:", theta)
    plt.plot(history["mse"])
    plt.title("MSE over epochs")
    plt.xlabel("epoch")
    plt.ylabel("mse")
    plt.show()
