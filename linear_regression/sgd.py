"""
Mini-batch / stochastic gradient descent implementation for linear regression.

Created by: Aditya Singh
Date: 2025-11-03

API:
 - fit_sgd(X, y, lr=0.01, epochs=100, batch_size=32, shuffle=True)

Returns (theta, history) similar to batch implementation.
"""
from typing import Tuple, Dict
import numpy as np
from .gradient_descent import add_intercept


def fit_sgd(X, y, lr: float = 0.01, epochs: int = 100, batch_size: int = 32, shuffle: bool = True) -> Tuple[np.ndarray, Dict]:
    Xb = add_intercept(X)
    y = np.asarray(y, dtype=float)
    n_samples, n_features = Xb.shape
    theta = np.zeros(n_features, dtype=float)
    history = {"mse": []}

    for epoch in range(epochs):
        if shuffle:
            idx = np.random.permutation(n_samples)
            Xb_shuf = Xb[idx]
            y_shuf = y[idx]
        else:
            Xb_shuf = Xb
            y_shuf = y

        for start in range(0, n_samples, batch_size):
            end = start + batch_size
            xb = Xb_shuf[start:end]
            yb = y_shuf[start:end]
            preds = xb.dot(theta)
            error = preds - yb
            grad = (2.0 / xb.shape[0]) * xb.T.dot(error)
            theta -= lr * grad

        # epoch mse
        preds_all = Xb.dot(theta)
        mse = float(((preds_all - y) ** 2).mean())
        history["mse"].append(mse)

    return theta, history


if __name__ == "__main__":
    rng = np.random.RandomState(0)
    X = rng.rand(200) * 10
    y = 1.0 + 2.0 * X + rng.randn(200) * 0.5
    theta, history = fit_sgd(X, y, lr=0.001, epochs=200, batch_size=16)
    print("theta:", theta)
