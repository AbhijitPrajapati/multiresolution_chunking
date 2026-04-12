from chromadb import PersistentClient
import json
from src.retrieval import retrieve_chunks
from src.metrics import Evaluator
from src.constants import (
    SIMILARITY_THRESHOLDS,
    EVAL_K,
    EVAL_LENGTH_THRESHOLD,
    EVAL_RATIO_THRESHOLD,
    EVAL_N_CHUNKS,
)

client = PersistentClient("chunks")


def main():
    with open("data/prompts/prompts.json", errors="ignore") as f:
        data = json.load(f)
    queries = []
    targets = []
    for p in data:
        queries.append(p["question"])
        targets.append(p["evidence"])

    evaluator = Evaluator(targets, EVAL_K, EVAL_RATIO_THRESHOLD, EVAL_LENGTH_THRESHOLD)

    fixed = client.get_collection("fixed_length")
    sentence_based = client.get_collection("sentence_based")
    semantic = client.get_collection("semantic")

    fixed_chunks = retrieve_chunks(queries, EVAL_N_CHUNKS, fixed)
    sentence_based_chunks = retrieve_chunks(queries, EVAL_N_CHUNKS, sentence_based)
    semantic_chunks = {
        thresh: retrieve_chunks(
            queries,
            EVAL_N_CHUNKS,
            semantic,
            md_filter={"similarity_threshold": thresh},
        )
        for thresh in SIMILARITY_THRESHOLDS
    }
    multiresolution_chunks = retrieve_chunks(queries, EVAL_N_CHUNKS, semantic)

    results = {
        "fixed_length": evaluator.get_metrics(fixed_chunks),  # type: ignore
        "sentence_based": evaluator.get_metrics(sentence_based_chunks),  # type: ignore
        "semantic": {k: evaluator.get_metrics(v) for k, v in semantic_chunks.items()},  # type: ignore
        "multiresolution": evaluator.get_metrics(multiresolution_chunks),  # type: ignore
    }

    with open("results/results.json", "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()
