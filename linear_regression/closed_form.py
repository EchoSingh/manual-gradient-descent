"""
Closed-form solution for linear regression using the normal equation.

Created by: Aditya Singh
Date: 2025-11-03

API:
 - solve_normal_equation(X, y) -> theta

Note: For numerical stability we use the pseudo-inverse.
"""
import numpy as np
from .gradient_descent import add_intercept


def solve_normal_equation(X, y):
    """Compute theta using the normal equation (closed-form).

    Args:
        X: (n_samples,) or (n_samples, n_features)
        y: (n_samples,)

    Returns:
        theta: (n_features+1,)
    """
    Xb = add_intercept(X)
    y = np.asarray(y, dtype=float)
    # theta = (X^T X)^(-1) X^T y ; use pinv for stability
    theta = np.linalg.pinv(Xb.T.dot(Xb)).dot(Xb.T).dot(y)
    return theta


if __name__ == "__main__":
    # quick sanity check
    rng = np.random.RandomState(0)
    X = rng.rand(50) * 10
    y = 4.0 + 1.5 * X + rng.randn(50) * 0.1
    theta = solve_normal_equation(X, y)
    print("theta:", theta)
