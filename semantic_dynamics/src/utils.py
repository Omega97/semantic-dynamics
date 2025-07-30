from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer


def sliding_window_embeddings(
    text: str,
    window_size: int,
    embedding_model: SentenceTransformer,
    stride: int = 1,
) -> np.ndarray:
    """
    Compute sentence-level embeddings for every sliding window of words
    in *text*.

    Parameters
    ----------
    text : str
        The document to embed.
    window_size : int
        Number of space-separated tokens per window.
    embedding_model : SentenceTransformer
        A pre-loaded model (e.g., SentenceTransformer("all-MiniLM-L6-v2")).
    stride : int, optional
        Step size between windows. 1 = fully overlapping,
        window_size = non-overlapping. Defaults to 1.

    Returns
    -------
    np.ndarray
        Array with shape (num_windows, embedding_dim).
    """
    if window_size <= 0:
        raise ValueError("window_size must be > 0")
    if stride <= 0:
        raise ValueError("stride must be > 0")

    # Basic whitespace tokenisation (replace with nltk or spaCy if needed)
    tokens: List[str] = text.split()
    num_tokens = len(tokens)

    if num_tokens < window_size:
        raise ValueError(
            f"Text is shorter ({num_tokens} tokens) than window_size={window_size}"
        )

    # Prepare list of window strings
    windows: List[str] = [
        " ".join(tokens[i : i + window_size])
        for i in range(0, num_tokens - window_size + 1, stride)
    ]

    # Batch-encode for efficiency
    embeddings = embedding_model.encode(
        windows, convert_to_numpy=True, show_progress_bar=False
    )

    return embeddings
