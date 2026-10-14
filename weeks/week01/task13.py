import matplotlib.pyplot as plt
import numpy as np


def main() -> int:
    rng = np.random.default_rng(42)

    sample_count = 10000
    uniform_samples = rng.uniform(0.0, 1.0, sample_count)
    normal_samples = rng.normal(0.0, 1.0, sample_count)
    exponential_samples = rng.exponential(1.0, sample_count)

    print(
        f"Uniform: mean={uniform_samples.mean():.4f}, std={uniform_samples.std():.4f}, min={uniform_samples.min():.4f}, max={uniform_samples.max():.4f}"
    )
    print(
        f"Normal: mean={normal_samples.mean():.4f}, std={normal_samples.std():.4f}, min={normal_samples.min():.4f}, max={normal_samples.max():.4f}"
    )
    print(
        f"Exponential: mean={exponential_samples.mean():.4f}, std={exponential_samples.std():.4f}, min={exponential_samples.min():.4f}, max={exponential_samples.max():.4f}"
    )

    figure, (axis_unif, axis_norm, axis_exp) = plt.subplots(1, 3, figsize=(12, 4))
    axis_unif.hist(uniform_samples, bins=30)
    axis_unif.set_title("Uniform distribution")
    axis_unif.set_xlabel("value")
    axis_unif.set_ylabel("count")
    axis_unif.grid(True)

    axis_norm.hist(normal_samples, bins=30)
    axis_norm.set_title("Normal distribution")
    axis_norm.set_xlabel("value")
    axis_norm.set_ylabel("count")
    axis_norm.grid(True)

    axis_exp.hist(exponential_samples, bins=30)
    axis_exp.set_title("Exponential distribution")
    axis_exp.set_xlabel("value")
    axis_exp.set_ylabel("count")
    axis_exp.grid(True)
    figure.tight_layout()

    plt.show()

    return 0


if __name__ == "__main__":
    main()
