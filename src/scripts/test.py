import numpy as np
from src.chunking import semantic_chunking
from tqdm import tqdm
import json


def load_papers():
    for i in tqdm(range(10)):
        fn = f"p{i}"
        with open(f"data/processed/{fn}.json") as f:
            data = json.load(f)
        yield data, fn


# chunk semantically with different similarity thresholds to see if high std subsides or not, as well as chunk per document count
cpd = []
chunk_lens = []
for data, fn in load_papers():
    chunks, section_names = semantic_chunking(data["content"])
    cpd.append(len(chunks))
    chunk_lens.extend([len(c) for c in chunks])

print(np.mean(chunk_lens), np.std(chunk_lens), np.mean(cpd))


# aim for 1600-1700 chars per chunk

# Semantic:
# thesh = 0.5
# Mean: 1212.3 STD: 1755.4 CPD: 57.0
# thesh = 0.6
# Mean: 570.6 STD: 617.7 CPD: 149.5
