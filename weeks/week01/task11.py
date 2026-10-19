import matplotlib.pyplot as plt
import numpy as np


def main() -> int:
    rng = np.random.default_rng(42)
    slope = 2.0
    intercept = 1.0
    points = 50
    noise = rng.normal(0.0, 2.0, points)
    x = np.linspace(0, 10, points)
    y = slope * x + intercept + noise

    print(f"Generated {points} points for y = {slope}*x + {intercept} + noise")
    print(f"x range: [{x.min():.2f}, {x.max():.2f}]")
    print(f"y range: [{y.min():.2f}, {y.max():.2f}]")

    figure, axis = plt.subplots()
    axis.scatter(x, y)

    axis.set_title("Synthetic linear data")
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.grid(True)
    figure.tight_layout()

    plt.show()

    return 0
