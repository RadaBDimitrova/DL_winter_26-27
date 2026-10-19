import numpy as np
import torch


def main() -> int:
    array = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    tensor = torch.from_numpy(array)

    print("NumPy array:")
    print(array)
    print(f"dtype={array.dtype}, shape={array.shape}")
    print()

    print("PyTorch tensor:")
    print(tensor)
    print(f"dtype={tensor.dtype}, shape={tensor.shape}, device={tensor.device}")
    print()

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    tensor = tensor.to(device)

    print(f"Selected device: {device}")
    print(f"Tensor moved to device: {tensor.device}, dtype={tensor.dtype}")

    return 0
