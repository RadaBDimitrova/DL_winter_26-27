import matplotlib.pyplot as plt
import numpy as np
import torch

from dl_lib.activations import functional


def main() -> int:
    x = np.linspace(-10.0, 10.0, 201)
    y = functional.sigmoid(x)

    print(f"sigmoid(-10) = {functional.sigmoid(np.array([-10.0]))[0]:.6f}")
    print(f"sigmoid(0)   = {functional.sigmoid(np.array([0.0]))[0]:.6f}")
    print(f"sigmoid(10)  = {functional.sigmoid(np.array([10.0]))[0]:.6f}")
    print(f"Output range: [{y.min():.6f}, {y.max():.6f}]")

    torch_y = torch.sigmoid(torch.from_numpy(x)).numpy()
    print(f"torch.sigmoid: {np.allclose(y, torch_y)}")

    figure, axis = plt.subplots()
    axis.plot(x, y, label="sigmoid(x)")

    axis.set_title("Sigmoid over [-10, 10]")
    axis.set_xlabel("x")
    axis.set_ylabel("sigmoid(x)")
    axis.legend()
    axis.grid(True)
    figure.tight_layout()
    plt.show()

    return 0
