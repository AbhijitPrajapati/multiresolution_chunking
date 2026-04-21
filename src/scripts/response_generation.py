import json
from src.nlp import generate
from src.vector_store import retrieve_chunks
from src.constants import EVAL_N_CHUNKS


def build_prompt(questions, contexts):
    


def main():
    with open("data/prompts/prompts.json", errors="ignore") as f:
        data = json.load(f)
    ids, questions = map(list, zip(*((p["id"], p["question"]) for p in data)))
    contexts_dict = {
        m: retrieve_chunks(questions, EVAL_N_CHUNKS, m)
        for m in ["fixed_length", "sentence_based", "semantic"]
    }
    responses = generate(prompts)
    dump = [{"id": i, "response": r} for i, r in zip(ids, responses)]
    with open("results/llm_responses.json", "w") as f:
        json.dump(dump, f)


if __name__ == "__main__":
    main()
