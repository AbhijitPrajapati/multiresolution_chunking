from chromadb import PersistentClient
import json
from tqdm import tqdm
from src.chunking import (
    fixed_length_chunking,
    semantic_chunking,
    sentence_based_chunking,
)
from src.vector_store import store_chunks


client = PersistentClient("chunks")


def load_papers():
    for i in tqdm(range(10)):
        fn = f"p{i}"
        with open(f"data/processed/{fn}.json") as f:
            data = json.load(f)
        yield data, fn


def main():
    for data, fn in load_papers():
        store_chunks(
            "fixed_length", fn, data["title"], *fixed_length_chunking(data["content"])
        )
        store_chunks(
            "sentence_based",
            fn,
            data["title"],
            *sentence_based_chunking(data["content"]),
        )
        store_chunks(
            "semantic",
            fn,
            data["title"],
            *semantic_chunking(data["content"]),
        )


if __name__ == "__main__":
    main()
