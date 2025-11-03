# AAT: Manual Gradient Descent for Linear Regression

This small project implements batch gradient descent for linear regression from scratch (no scikit-learn).

Files:
- `linear_regression/gradient_descent.py` — core implementation (fit/predict/mse).
- `demo.py` — generate synthetic data, fit model, save plots and print results.
- `tests/test_gradient_descent.py` — pytest unit test.
- `requirements.txt` — Python dependencies.

How to run:

1. Create a Python virtual environment and install requirements:

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1; python -m pip install -r requirements.txt
```

2. Run the demo to fit and visualize the model:

```powershell
python demo.py
```

3. Run tests:

```powershell
python -m pytest -q
```

