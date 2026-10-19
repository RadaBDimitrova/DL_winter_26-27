import numpy as np


def main() -> int:
    y_true = np.array([3, 5, 7, 9])
    y_pred = np.array([2, 6, 8, 7])

    errs = y_pred - y_true
    abs_errs = np.abs(errs)
    sqrd_errs = errs**2

    mean_err = errs.mean()
    mae = abs_errs.mean()
    mse = sqrd_errs.mean()
    rmse = np.sqrt(mse)

    print(f"y_true:          {y_true}")
    print(f"y_pred:          {y_pred}")
    print(f"errors:          {errs}")
    print(f"absolute errors: {abs_errs}")
    print(f"squared errors:  {sqrd_errs}")
    print()
    print(f"mean error: {mean_err:.4f}")
    print(f"MAE (mean absolute error): {mae:.4f}")
    print(f"MSE (mean squared error):  {mse:.4f}")
    print(f"RMSE (root mean squared error): {rmse:.4f}")

    return 0
