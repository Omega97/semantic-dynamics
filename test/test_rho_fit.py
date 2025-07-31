"""
Density Estimation Test
"""
import numpy as np
from pathlib import Path
from semantic_dynamics.src.density_estimation import HybridDensityEstimator
from semantic_dynamics.src.utils import find_npy_files


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


def main(n_components: int = 5, gmm_components: int = 3) -> None:
    # ------------------------------------------------------------------ #
    # 1) Locate one .npy file with embeddings
    # ------------------------------------------------------------------ #
    npy_files = find_npy_files(DATA_DIR)
    if not npy_files:
        raise FileNotFoundError(f"No .npy files found under {DATA_DIR}")

    npy_path = npy_files[0]
    print(f"\nProcessing {npy_path.name}...")

    # ------------------------------------------------------------------ #
    # 2) Load embeddings
    # ------------------------------------------------------------------ #
    embeddings = np.load(npy_path)           # (n, d)
    n, d = embeddings.shape
    print(f"Loaded {n} embeddings of dimension {d}")

    # ------------------------------------------------------------------ #
    # 3) Fit HybridDensityEstimator
    # ------------------------------------------------------------------ #
    estimator = HybridDensityEstimator(
        k=n_components,
        gmm_components=gmm_components,
    ).fit(embeddings)

    # ------------------------------------------------------------------ #
    # 4) Evaluate likelihood on a small slice
    # ------------------------------------------------------------------ #
    test_points = embeddings[100:105]        # (5, d)
    avg_ll = estimator.score(test_points)
    print(f"Average log-likelihood on 5 test points: {avg_ll:.3f}")

    # ------------------------------------------------------------------ #
    # 5) Inspect learned parameters
    # ------------------------------------------------------------------ #
    print("Model summary:", estimator.get_params())

    # ------------------------------------------------------------------ #
    # 6) Draw a few synthetic samples (optional sanity check)
    # ------------------------------------------------------------------ #
    x_samples, _ = estimator.sample(3)
    print("Three generated samples (original space):")
    print(x_samples)


if __name__ == "__main__":
    main(n_components=5, gmm_components=3)
