import json
from tqdm import tqdm


def load_papers():
    for i in tqdm(range(10)):
        fn = f"p{i}"
        with open(f"data/processed/{fn}.json") as f:
            data = json.load(f)
        yield data, fn
