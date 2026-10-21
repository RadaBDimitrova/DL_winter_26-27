import numpy as np
import torch

from dl_lib.activations import functional


def main() -> int:
    scores = np.array([1.0, 2.0, 3.0])
    probabilities = functional.softmax(scores)
    print(f"scores: {scores}")
    print(f"softmax: {probabilities}")
    print(f"sum of probabilities: {probabilities.sum():.4f}")
    print()

    large_scores = np.array([1000.0, 1001.0, 1002.0])
    # numerically unstable naive softmax
    with np.errstate(over="ignore", invalid="ignore"):
        naive = np.exp(large_scores) / np.sum(np.exp(large_scores))
    print(f"Large scores: {large_scores}")
    print(f"Naive softmax (unstable): {naive}")
    print(f"Stable softmax: {functional.softmax(large_scores)}")

    torch_probabilities = torch.softmax(torch.from_numpy(scores), dim=-1).numpy()
    print(f"torch.softmax: {np.allclose(probabilities, torch_probabilities)}")

    return 0
