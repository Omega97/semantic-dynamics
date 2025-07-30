"""
Simple script: compute sliding‑window embeddings for each .txt file in the
*data/* directory using the `sliding_window_embeddings` utility.  Mirrors the
formatting style of *test_embedding.py*.

Run:
    python test_sliding_window.py

(Optional flags could be added later if needed.)
"""
import numpy as np
import sys
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Adjust the import path if your project structure differs
try:
    from semantic_dynamics.src.utils import sliding_window_embeddings
except ModuleNotFoundError as e:  # fallback for direct execution in flat tree
    print("[ERROR] Cannot import sliding_window_embeddings. Ensure PYTHONPATH is set.", file=sys.stderr)
    raise e

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
WINDOW_SIZE = 50  # words per window
STRIDE = 3        # step size (1 = fully overlapping, WINDOW_SIZE = non‑overlap)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


def find_file_names() -> list[Path]:
    txt_files = sorted(DATA_DIR.glob("*.txt"))
    if not txt_files:
        print(f"[ERROR] No .txt files found in {DATA_DIR}", file=sys.stderr)
        sys.exit(1)
    return txt_files


def main() -> None:
    print("Loading model...")
    model = SentenceTransformer(DEFAULT_MODEL_NAME)

    file_names = find_file_names()
    file_names = file_names[:1]

    for text_path in file_names:
        print(f"\n[File] {text_path.name}")
        text = text_path.read_text(encoding="utf-8")

        print("Computing sliding‑window embeddings...")
        emb_matrix = sliding_window_embeddings(
            text=text,
            window_size=WINDOW_SIZE,
            embedding_model=model,
            stride=STRIDE,
        )

        out_path = text_path.with_suffix(".npy")  # e.g. data/article.txt → data/article.npy
        np.save(out_path, emb_matrix)

        print("→ Matrix shape:", emb_matrix.shape)


if __name__ == "__main__":
    main()
