import numpy as np
from linear_regression.gradient_descent import fit, predict
from linear_regression.closed_form import solve_normal_equation
from linear_regression.sgd import fit_sgd


def test_gradient_descent_recovers_linear_params():
    # y = 2 + 3*x
    rng = np.random.RandomState(0)
    X = rng.rand(200) * 10
    y = 2.0 + 3.0 * X + rng.randn(200) * 0.5

    theta, history = fit(X, y, lr=0.001, epochs=3000, verbose=False)

    # theta[0] ~ intercept, theta[1] ~ slope
    assert abs(theta[0] - 2.0) < 0.5
    assert abs(theta[1] - 3.0) < 0.2


def test_closed_form_matches_true_params():
    rng = np.random.RandomState(1)
    X = rng.rand(100) * 5
    y = 1.5 + 2.5 * X + rng.randn(100) * 0.1
    theta = solve_normal_equation(X, y)
    # close to true params
    assert abs(theta[0] - 1.5) < 0.2
    assert abs(theta[1] - 2.5) < 0.05


def test_sgd_recovers_linear_params():
    rng = np.random.RandomState(2)
    X = rng.rand(300) * 10
    y = -1.0 + 0.75 * X + rng.randn(300) * 0.5
    theta, history = fit_sgd(X, y, lr=0.001, epochs=300, batch_size=32, shuffle=True)
    assert abs(theta[0] - (-1.0)) < 0.5
    assert abs(theta[1] - 0.75) < 0.2
