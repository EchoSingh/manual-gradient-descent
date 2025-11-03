"""Compare closed-form, batch gradient descent, and SGD on synthetic data.

Created by: Aditya Singh
Date: 2025-11-03

This script prints learned parameters and MSE for each method and saves
a plot `compare_methods.png` showing the data and the three fitted lines.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from linear_regression.closed_form import solve_normal_equation
from linear_regression.gradient_descent import fit, predict, compute_mse
from linear_regression.sgd import fit_sgd


def make_data(n=200, noise=0.5, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.rand(n) * 10
    y = 2.0 + 3.0 * X + rng.randn(n) * noise
    return X, y


def main():
    X, y = make_data(n=200, noise=0.5, seed=0)

    theta_cf = solve_normal_equation(X, y)

    theta_gd, history_gd = fit(X, y, lr=0.001, epochs=3000, verbose=False)

    theta_sgd, history_sgd = fit_sgd(X, y, lr=0.001, epochs=300, batch_size=32, shuffle=True)

    mse_cf = compute_mse(X, y, theta_cf)
    mse_gd = compute_mse(X, y, theta_gd)
    mse_sgd = compute_mse(X, y, theta_sgd)

    print("Closed-form theta:", np.round(theta_cf, 6), "MSE:", round(mse_cf, 6))
    print("Batch GD theta:   ", np.round(theta_gd, 6), "MSE:", round(mse_gd, 6))
    print("SGD theta:        ", np.round(theta_sgd, 6), "MSE:", round(mse_sgd, 6))

    # Plot data and fits
    xs = np.linspace(X.min(), X.max(), 200)
    ys_cf = predict(xs, theta_cf)
    ys_gd = predict(xs, theta_gd)
    ys_sgd = predict(xs, theta_sgd)

    plt.figure(figsize=(8, 6))
    plt.scatter(X, y, s=20, alpha=0.5, label="data")
    plt.plot(xs, ys_cf, label="closed-form", color="tab:blue")
    plt.plot(xs, ys_gd, label="batch GD", color="tab:orange", linestyle="--")
    plt.plot(xs, ys_sgd, label="SGD", color="tab:green", linestyle=":")
    plt.legend()
    plt.title("Linear regression: closed-form vs batch GD vs SGD")
    plt.xlabel("X")
    plt.ylabel("y")

    out_path = os.path.abspath("compare_methods.png")
    plt.savefig(out_path)
    print(f"Saved comparison plot to: {out_path}")


if __name__ == "__main__":
    main()
