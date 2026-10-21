import matplotlib.pyplot as plt
import numpy as np
import torch

from dl_lib.activations import functional


def main() -> int:
    x = np.linspace(-10.0, 10.0, 201)
    activations = {
        "sigmoid": functional.sigmoid(x),
        "relu": functional.relu(x),
        "tanh": functional.tanh(x),
    }

    for name, values in activations.items():
        print(f"{name} range:".ljust(15) + f"[{values.min():.4f}, {values.max():.4f}]")
    print()

    for sample in (-5.0, 0.0, 5.0):
        point = np.array([sample])
        print(
            f"At x={sample:.0f}:".ljust(9)
            + f"sigmoid={functional.sigmoid(point)[0]:.4f}, "
            f"relu={functional.relu(point)[0]:.4f}, "
            f"tanh={functional.tanh(point)[0]:.4f}"
        )

    torch_x = torch.from_numpy(x)
    matches = (
        np.allclose(activations["sigmoid"], torch.sigmoid(torch_x).numpy())
        and np.allclose(activations["relu"], torch.relu(torch_x).numpy())
        and np.allclose(activations["tanh"], torch.tanh(torch_x).numpy())
    )
    print(f"torch.sigmoid/relu/tanh: {matches}")

    figure, axis = plt.subplots()
    for name, values in activations.items():
        axis.plot(x, values, label=name)

    axis.set_title("Sigmoid, ReLU, and tanh")
    axis.set_xlabel("x")
    axis.set_ylabel("activation(x)")
    axis.legend()
    axis.grid(True)
    figure.tight_layout()
    plt.show()

    return 0
