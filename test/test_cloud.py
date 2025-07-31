"""
Scatter plot the cloud
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from pathlib import Path
from semantic_dynamics.src.utils import find_npy_files


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


def scatter_plot(x: np.ndarray, y: np.ndarray, out_path: Path, dim_ids=(0, 1)) -> None:
    """Create and save a scatter plot of the first two columns."""
    plt.figure()
    plt.scatter(x[:100], y[:100], s=10, c='b')
    plt.scatter(x, y, s=10, alpha=0.2, c='b')  # default color, avoids specifying colors explicitly
    plt.title(f"2-D Scatter of Embeddings for {out_path.stem}")
    plt.xlabel(f"Embedding dim {dim_ids[0]}")
    plt.ylabel(f"Embedding dim {dim_ids[1]}")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main(n_components=100, dim_ids=(0, 1)):

    # Find files
    npy_files = find_npy_files(DATA_DIR)

    for npy_path in npy_files:

        # Load embedding vectors
        print(f"Processing {npy_path.name} …")
        embeddings = np.load(npy_path)

        # Apply PCA to reduce to first components
        pca = PCA(n_components=n_components)
        components = pca.fit_transform(embeddings)

        # plot explained_variance_
        # print(f"→ PCA eigenvalues: {pca.explained_variance_}")
        # plt.title("Expected Variance")
        # plt.plot(pca.explained_variance_)
        # plt.show()

        # Split into x and y for plotting
        x, y = components.T[dim_ids[0]], components.T[dim_ids[1]]

        out_png = npy_path.parent / f"{npy_path.stem}_{dim_ids[0]}_{dim_ids[1]}.png"
        scatter_plot(x=x, y=y, out_path=out_png, dim_ids=dim_ids)
        print(f"→ Saved plot to {out_png}")


if __name__ == "__main__":
    main()
