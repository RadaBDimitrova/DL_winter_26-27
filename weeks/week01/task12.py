import matplotlib.pyplot as plt
import numpy as np


def main() -> int:
    rng = np.random.default_rng(42)
    x = np.linspace(0, 2 * np.pi, 100)
    sin = np.sin(x)
    cos = np.cos(x)
    sample_count = 1000
    samples = rng.normal(0.0, 1.0, sample_count)

    print(f"sin(x) range: [{sin.min():.4f}, {sin.max():.4f}]")
    print(f"Histogram sample count: {sample_count}")
    figure, (axis_plot, axis_hist) = plt.subplots(1, 2, figsize=(12, 4))

    axis_plot.plot(x, sin, label="sin(x)")
    axis_plot.plot(x, cos, label="cos(x)")
    axis_plot.set_title("Line plot")
    axis_plot.set_xlabel("x")
    axis_plot.set_ylabel("y")
    axis_plot.legend()
    axis_plot.grid(True)

    axis_hist.hist(samples, bins=30)
    axis_hist.set_title("Histogram of normal samples")
    axis_hist.set_xlabel("value")
    axis_hist.set_ylabel("count")
    axis_hist.grid(True)
    figure.tight_layout()

    plt.show()

    return 0
