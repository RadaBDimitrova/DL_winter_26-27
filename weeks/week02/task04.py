import matplotlib.pyplot as plt
import numpy as np

from dl_lib.losses import functional


def main() -> int:
    mae_values = []
    mse_values = []
    rmse_values = []
    fixed_errors = np.array([0.5, 1.0, 1.0])
    growing_errors = np.linspace(0.0, 10.0, 101)

    for growing in growing_errors:
        y_pred = np.append(fixed_errors, growing)
        y_true = np.zeros_like(y_pred)
        mae_values.append(functional.mae(y_true, y_pred))
        mse_values.append(functional.mse(y_true, y_pred))
        rmse_values.append(functional.rmse(y_true, y_pred))

    for sample in (1.0, 5.0, 10.0):
        y_pred = np.append(fixed_errors, sample)
        y_true = np.zeros_like(y_pred)
        print(
            f"Growing error={sample:.1f}: "
            f"MAE={functional.mae(y_true, y_pred):.4f}, "
            f"MSE={functional.mse(y_true, y_pred):.4f}, "
            f"RMSE={functional.rmse(y_true, y_pred):.4f}"
        )

    figure, axis = plt.subplots()
    axis.plot(growing_errors, mae_values, label="MAE")
    axis.plot(growing_errors, mse_values, label="MSE")
    axis.plot(growing_errors, rmse_values, label="RMSE")

    axis.set_title("MSE, MAE, and RMSE growth as one prediction error increases")
    axis.set_xlabel("the growing error, with three other errors held fixed ")
    axis.set_ylabel("loss value")
    axis.legend()
    axis.grid(True)
    figure.tight_layout()
    plt.show()

    return 0
