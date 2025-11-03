"""Demo script: generate data, fit gradient descent, plot results."""
import os
import numpy as np
import matplotlib.pyplot as plt
from linear_regression.gradient_descent import fit, predict, compute_mse


def make_data(n=100, noise=2.0, seed=42):
    rng = np.random.RandomState(seed)
    X = rng.rand(n) * 10
    y = 3.0 * X + 2.0 + rng.randn(n) * noise
    return X, y


def main():
    X, y = make_data(n=100, noise=2.0)
    theta, history = fit(X, y, lr=0.001, epochs=5000, verbose=False)

    print("Learned parameters:", theta)
    mse = compute_mse(X, y, theta)
    print(f"Final MSE: {mse:.6f}")

    # Plot data and fit
    xs = np.linspace(X.min(), X.max(), 100)
    ys = predict(xs, theta)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].scatter(X, y, alpha=0.6, label="data")
    axes[0].plot(xs, ys, color="red", label="fit")
    axes[0].set_title("Data and linear fit")
    axes[0].legend()

    axes[1].plot(history["mse"])
    axes[1].set_title("MSE over epochs")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("mse")

    out_dir = os.path.abspath(".")
    fig_path = os.path.join(out_dir, "fit_and_loss.png")
    fig.savefig(fig_path)
    print(f"Saved plot to: {fig_path}")
    plt.show()


if __name__ == "__main__":
    main()
