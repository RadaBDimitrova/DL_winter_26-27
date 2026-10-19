import numpy as np
import torch

from dl_lib.losses import functional


def main() -> int:
    y_true = np.array([3.0, 5.0, 7.0, 9.0])
    y_pred = np.array([2.0, 6.0, 8.0, 7.0])
    res = functional.mae(y_true, y_pred)
    torch_res = torch.nn.functional.l1_loss(
        torch.from_numpy(y_pred), torch.from_numpy(y_true)
    ).item()

    print(f"y_true: {y_true}")
    print(f"y_pred: {y_pred}")
    print(f"MAE: {res:.4f}")
    print(f"MAE (torch l1_loss): {torch_res:.4f}")

    return 0
