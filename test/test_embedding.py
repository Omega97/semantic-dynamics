"""
Simple script: download a SentenceTransformer model and embed a text file.
"""
import sys
from pathlib import Path
from sentence_transformers import SentenceTransformer


DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


def find_file_names() -> list:
    txt_files = sorted(DATA_DIR.glob("*.txt"))
    if not txt_files:
        print(f"[ERROR] No .txt files found in {DATA_DIR}", file=sys.stderr)
        sys.exit(1)
    return txt_files


def main():

    print(f"Loading model...")
    model = SentenceTransformer(DEFAULT_MODEL_NAME)

    for text_path in find_file_names():

        print(f"Reading '{text_path}'...")
        text = text_path.read_text(encoding="utf-8")

        print("Encoding...")
        embedding = model.encode(text)
        print(f'{embedding[:5]}...')


if __name__ == "__main__":
    main()
